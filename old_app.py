from flask import Flask, render_template, jsonify, request, flash, redirect, url_for, Response, send_file
import json
import os
from weasyprint import HTML
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Email
import smtplib
from email.mime.text import MIMEText
from flask_cors import CORS  # Import Flask-CORS
import logging
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Enable CORS for all routes
CORS(app)





###############################
# thermodynamic modules
###############################
## thermodynamic modules
import primer3
import RNA
import plotly.graph_objs as go
import plotly.io as pio
from Bio import Seq

@app.route('/calculate_tm', methods=['POST', 'GET'])
def calculate_tm():
    if request.method =='GET':
        sequence = request.args.get('sequence', '').upper()
    else:
        data = request.get_json()

        # Extract the oligonucleotide sequence from the POST request
        sequence = data.get('sequence')

    # Validate the input
    if not sequence:
        return jsonify({'error': 'Sequence is required.'}), 400
    
    try:
        # Calculate Tm using primer3
        tm = primer3.calc_tm(sequence.upper())
        # Return the result as JSON
        return jsonify({
            'sequence': sequence,
            'tm': tm
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/calculate_hairpin', methods=['GET'])
def check_hairpin():
    sequence = request.args.get('sequence', '').upper()
    hairpin = primer3.calcHairpin(sequence)

    # If hairpin deltaG is positive, it indicates no stable hairpin
    if hairpin.dg >= 0:
        result = "None"
        hairpin_structure = "." * len(sequence)  # No hairpin structure
    else:
        result = hairpin.dg
        
        # Find the hairpin structure visualization
        hairpin_structure = visualize_hairpin(sequence)

    return jsonify({"hairpin_dg": result, "hairpin_structure": hairpin_structure})

def visualize_hairpin(sequence):
    # Create a structure string initialized with '.'
    structure = ['.'] * len(sequence)
    
    # A simple algorithm to find hairpin structures by checking for palindromes
    seq_length = len(sequence)

    for i in range(seq_length):
        for j in range(i + 4, seq_length):  # Minimum loop size is 3
            # Check for a palindromic sequence
            if is_hairpin(sequence, i, j):
                # Mark the matching bases
                structure[i] = '('
                structure[j] = ')'
                # Optionally break after first found hairpin for simplicity
                break

    return ''.join(structure)

def is_hairpin(sequence, start, end):
    """Check if the sequence from start to end forms a hairpin."""
    if end - start < 3:
        return False  # Minimum hairpin structure length

    # Check for complementary bases (A-T, C-G)
    for k in range((end - start + 1) // 2):
        base1 = sequence[start + k]
        base2 = sequence[end - k]
        if not (base1 == 'A' and base2 == 'T' or base1 == 'T' and base2 == 'A' or
                base1 == 'C' and base2 == 'G' or base1 == 'G' and base2 == 'C'):
            return False
            
    return True



@app.route('/design_oligos', methods=['POST'])
def design_oligo():
    # Get user input from the form
    sequence = request.form.get('sequence')
    min_gc = int(request.form.get('min_gc'))
    max_gc = int(request.form.get('max_gc'))
    primer_len = int(request.form.get('primer_len'))

    # Validate input
    if not sequence or len(sequence) < primer_len:
        return jsonify({'error': 'Invalid sequence or too short for primer length'}), 200
    
    # Primer3: Scan sequence for potential primers
    candidates = []
    for i in range(len(sequence) - primer_len + 1):
        oligo = sequence[i:i + primer_len].upper()
        gc_content = (oligo.count('G') + oligo.count('C')) / len(oligo) * 100
        tm = primer3.calc_tm(oligo.upper())
        
        # Only accept primers within GC content range
        if min_gc <= gc_content <= max_gc:
            candidates.append({
                'oligo': oligo,
                'gc_content': round(gc_content, 2),
                'tm': round(tm, 2),
                'position': i + 1
            })
    
    if candidates:
        return jsonify(candidates)
    else:
        return jsonify({'error': 'No suitable primers found'}), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_sequence():
    data = request.get_json()
    sequence = data['sequence'].upper() 
    naConcentration = float(data['sodium_concentration'])
    mgConcentration = float(data['magnesium_concentration'])
    minTemp = 20
    maxTemp = 90
    if "temperature_start" in data:
        minTemp = int(data['temperature_start'])
    if "temperature_end" in data:
        maxTemp = int(data['temperature_end'])

    # Generate the melt curve (temperatures vs fractions)
    temperatures, fractions = generate_melt_curve(sequence, naConcentration, mgConcentration, minTemp, maxTemp)

    # print(temperatures, fractions)
    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temperatures, y=fractions, mode='lines', name='Melt Curve'))
    fig.update_layout(title="Simulated DNA Melt Curve", xaxis_title="Temperature (°C)", yaxis_title="Fraction Double-Stranded")

    # print(fig)
    # Convert Plotly figure to JSON format
    plot_html = pio.to_html(fig, full_html=False)
    return jsonify({'plot_html': plot_html})

import math

# Constants
R = 1.987  # Gas constant in cal/(K*mol)

# Nearest-neighbor thermodynamic parameters for DNA duplexes (simplified)
nn_params = {
    'AA': (-7.9, -22.2), 'TT': (-7.9, -22.2),
    'AT': (-7.2, -20.4), 'TA': (-7.2, -21.3),
    'CA': (-8.5, -22.7), 'TG': (-8.5, -22.7),
    'GT': (-8.4, -22.4), 'AC': (-8.4, -22.4),
    'CT': (-7.8, -21.0), 'AG': (-7.8, -21.0),
    'GA': (-8.2, -22.2), 'TC': (-8.2, -22.2),
    'CG': (-10.6, -27.2), 'GC': (-9.8, -24.4),
    'GG': (-8.0, -19.9), 'CC': (-8.0, -19.9),
}

def calculate_fraction_double_stranded(sequence, temperature, na_concentration, mg_concentration, dna_concentration=50e-9):
    primer3_params = {
        'temp_c': temperature,  # Temperature in Celsius
        'mv_conc': na_concentration*1000,  # Sodium concentration in mM
        'dv_conc': mg_concentration*1000,  # Magnesium concentration in mM
        'dna_conc': dna_concentration*1e9
    }
    
    delta_g = primer3.calcHeterodimer(sequence, str(Seq.Seq(sequence).reverse_complement()), **primer3_params).dg

    # Convert temperature to Kelvin
    temperature_kelvin = temperature + 273.15

    # Calculate the equilibrium constant K_eq
    K_eq = math.exp(-delta_g / (R * temperature_kelvin))

    # Use DNA concentration directly (in M)
    K_eq_adjusted = K_eq * dna_concentration  # dna_concentration is already in M

    # Calculate fraction of DNA in double-helix form
    fraction_double_stranded = K_eq_adjusted / (1 + K_eq_adjusted)

    return fraction_double_stranded


def generate_melt_curve(sequence, na_concentration, mg_concentration, min_temp=20, max_temp=101):
    temperatures = list(range(min_temp, max_temp, 1))  # 20°C to 100°C
    fractions = [calculate_fraction_double_stranded(sequence, temp, na_concentration, mg_concentration) for temp in temperatures]
    return temperatures, fractions


def calculate_bound_concentration(primer_concentration, target_concentration, K_binding):
    # Calculate the quadratic equation terms
    a = 1
    b = -(primer_concentration + target_concentration + (1 / K_binding))
    c = primer_concentration * target_concentration
    # Solve the quadratic equation
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("Negative discriminant, no real solution for bound concentration.")
    bound_concentration = (-b - math.sqrt(discriminant)) / (2 * a)
    return bound_concentration
def equilibrium_distribution(primer_concentration, target_concentration, 
                             K_hairpin_primer, K_hairpin_target, K_binding):
    # Initialize random coil concentrations
    primer_random = primer_concentration / (1 + K_hairpin_primer)
    target_random = target_concentration / (1 + K_hairpin_target)
    bound = calculate_bound_concentration(primer_random, target_random, K_binding)
    print("asdfasdfasdf", target_concentration, target_random, bound, K_hairpin_primer, K_hairpin_target, K_binding)
    # Iteratively solve for equilibrium states
    for _ in range(100):  # Loop until convergence
        # Recalculate random coil concentrations accounting for binding
        primer_random = (primer_concentration - bound) / (1 + K_hairpin_primer)
        target_random = (target_concentration - bound) / (1 + K_hairpin_target)
        # Calculate hairpin concentrations based on updated random coil concentrations
        primer_hairpin = primer_concentration - primer_random - bound
        target_hairpin = target_concentration - target_random - bound
        # Calculate bound concentration
        bound = calculate_bound_concentration(primer_random, target_random, K_binding)
        # if bound >= min(target_concentration, primer_concentration):
        #     break
        print(">", _, target_concentration, target_random, bound, bound>target_concentration)
    
    primer_random = (primer_concentration - bound) / (1 + K_hairpin_primer)
    target_random = (target_concentration - bound) / (1 + K_hairpin_target)
    return {
        "primer_random": primer_random,
        "primer_hairpin": primer_hairpin,
        "target_random": target_random,
        "target_hairpin": target_hairpin,
        "bound": bound
    }

def calculate_binding_states(primer, target, sodium_concentration, magnesium_concentration, primer_concentration, target_concentration, temperature):
    primer3_params = {
        'temp_c': temperature,  # Temperature in Celsius
        'mv_conc': sodium_concentration*1000,  # Sodium concentration in mM
        'dv_conc': magnesium_concentration*1000,  # Magnesium concentration in mM
    }
    print(primer, primer_concentration, target_concentration, primer3_params)
    # Calculate the dG for hairpin formations
    hairpin_primer = primer3.calc_hairpin(primer, dna_conc=primer_concentration*1e9, **primer3_params)
    hairpin_target = primer3.calc_hairpin(target, dna_conc=target_concentration*1e9, **primer3_params)
    hairpin_primer_dG = hairpin_primer.dg
    hairpin_target_dG = hairpin_target.dg

    temperature_kelvin = temperature + 273.15

    # K_eq = math.exp(-delta_g / (R * temperature_kelvin))
    # Calculate K_hairpin
    if hairpin_primer.structure_found:
        K_hairpin_primer = math.exp(-hairpin_primer_dG / (R * temperature_kelvin))
    else:
        K_hairpin_primer = 0
    if hairpin_target.structure_found:
        K_hairpin_target = math.exp(-hairpin_target_dG / (R * temperature_kelvin))
    else:
        K_hairpin_target = 0

    heterodimer = primer3.calcHeterodimer(primer.upper(), target.upper(), dna_conc=(primer_concentration+target_concentration)*1e9, **primer3_params)
    binding_dG = heterodimer.dg  # dG is returned in cal/mol
    K_binding = math.exp(-binding_dG / (R * temperature_kelvin))
    # print("ASDFSDFAF", binding_dG, K_binding, sodium_concentration, magnesium_concentration)

    multi_state_eq = equilibrium_distribution(primer_concentration, target_concentration, K_hairpin_primer, K_hairpin_target, K_binding)
    # print(multi_state_eq)

    # Calculate the effective concentrations of the target in random coil and hairpin forms
    random_coil_target = target_concentration / (1 + K_hairpin_target)
    hairpin_target = (K_hairpin_target * target_concentration) / (1 + K_hairpin_target)

    # Similarly, for the primer:
    random_coil_primer = primer_concentration / (1 + K_hairpin_primer)
    hairpin_primer = (K_hairpin_primer * primer_concentration) / (1 + K_hairpin_primer)



    print(random_coil_primer, random_coil_target)
    print(hairpin_primer_dG, hairpin_target_dG, binding_dG)
    print(K_hairpin_primer, K_hairpin_target, K_binding)

    # return {
    #     "primer_random": primer_random,
    #     "primer_hairpin": primer_hairpin,
    #     "target_random": target_random,
    #     "target_hairpin": target_hairpin,
    #     "bound": bound
    # }


    hairpin_primer_percent = multi_state_eq['primer_hairpin']/primer_concentration * 100
    hairpin_target_percent = multi_state_eq['target_hairpin']/target_concentration * 100
    random_coil_primer_percent = multi_state_eq['primer_random']/primer_concentration * 100
    random_coil_target_percent = multi_state_eq['target_random']/target_concentration * 100
    binding_primer_percent = multi_state_eq['bound'] / primer_concentration * 100
    binding_target_percent = multi_state_eq['bound']  / target_concentration * 100
   

    return {
        "hairpin_primer_percent": hairpin_primer_percent,
        "binding_primer_percent": binding_primer_percent,
        "random_coil_primer_percent": random_coil_primer_percent,
        "hairpin_target_percent": hairpin_target_percent,
        "binding_target_percent": binding_target_percent,
        "random_coil_target_percent": random_coil_target_percent,
    }

@app.route('/calculate_binding', methods=['POST'])
def calculate_binding():
    import matplotlib.pyplot as plt
    import io
    import base64

    data = request.json
    if 'primer' in data:
        primer = data.get('primer')
    else:
        primer = data.get('sequence')
    if 'target' in data:
        target = data.get('target')
    else:
        target = data.get('target_sequence')
    sodium_concentration = data.get('sodium_concentration')
    magnesium_concentration = data.get('magnesium_concentration')
    if not sodium_concentration:
        sodium_concentration = 0.05
    if not magnesium_concentration:
        magnesium_concentration = 0
    sodium_concentration = float(sodium_concentration)
    magnesium_concentration = float(magnesium_concentration)

    primer_concentration = 1e-7
    target_concentration = 1e-9

    if "primer_concentration" in data:
        primer_concentration = float(data.get('primer_concentration'))
    elif "oligo_concentration" in data:
        primer_concentration = float(data.get('oligo_concentration'))
    if "target_concentration" in data:
        target_concentration = float(data.get('target_concentration'))
    else:
        target_concentration = 1e-10

    # print("AFASDFASDFA", primer_concentration, target_concentration)
    temperature = data.get('temperature')
    if not temperature:
        temperature = 60
    temperature = int(temperature)

    # Perform calculations (using Primer3 or ViennaRNA logic, adapted as discussed)
    binding_data = calculate_binding_states(primer, target, sodium_concentration, magnesium_concentration, primer_concentration, target_concentration, temperature)
    primer_hairpin_percent = binding_data['hairpin_primer_percent']  # Example calculation result
    primer_random_coil_percent = binding_data['random_coil_primer_percent']
    binding_percent_primer = binding_data['binding_primer_percent']
    primer_unaccounted = 100 - (primer_hairpin_percent + primer_random_coil_percent + binding_percent_primer)

    target_hairpin_percent = binding_data['hairpin_target_percent']
    target_random_coil_percent = binding_data['random_coil_target_percent']
    binding_percent_target = binding_data['binding_target_percent']
    target_unaccounted = 100 - (target_hairpin_percent + target_random_coil_percent + binding_percent_target)

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['Primer', 'Target']
    hairpin_data = [primer_hairpin_percent, target_hairpin_percent]
    random_coil_data = [primer_random_coil_percent, target_random_coil_percent]
    binding_data = [binding_percent_primer, binding_percent_target]
    unaccounted_data = [primer_unaccounted, target_unaccounted]

    # Stacked bars
    ax.bar(categories, hairpin_data, label='Hairpin', color='blue', alpha=0.6)
    ax.bar(categories, random_coil_data, bottom=hairpin_data, label='Random Coil', color='green', alpha=0.6)
    ax.bar(categories, binding_data, bottom=[hairpin + coil for hairpin, coil in zip(hairpin_data, random_coil_data)],
           label='Binding', color='orange', alpha=0.6)
    ax.bar(categories, unaccounted_data, bottom=[hairpin + coil + bind for hairpin, coil, bind in zip(hairpin_data, random_coil_data, binding_data)],
           label='Unaccounted', color='grey', alpha=0.6)

    ax.set_ylabel('Percentages')
    ax.set_title('Primer and Target Binding States')
    ax.legend()

    # Convert plot to a PNG image in base64 format
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Create an HTML table with the percentages
    table_html = f"""
    <table class="table table-bordered mt-4">
        <thead>
            <tr>
                <th>State</th>
                <th>Primer (%)</th>
                <th>Target (%)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Hairpin</td>
                <td>{primer_hairpin_percent}</td>
                <td>{target_hairpin_percent}</td>
            </tr>
            <tr>
                <td>Random Coil</td>
                <td>{primer_random_coil_percent}</td>
                <td>{target_random_coil_percent}</td>
            </tr>
            <tr>
                <td>Binding</td>
                <td>{binding_percent_primer}</td>
                <td>{binding_percent_target}</td>
            </tr>
            <tr>
                <td>Unaccounted</td>
                <td>{primer_unaccounted}</td>
                <td>{target_unaccounted}</td>
            </tr>
        </tbody>
    </table>
    """

    # Combine image and table for the response
    plot_html = f'<img src="data:image/png;base64,{image_base64}" class="img-fluid" alt="Result Plot"/>{table_html}'

    return jsonify({'plot_html': plot_html})

#################
# vienna rna test
#################
def calculate_free_energy(sequence, sodium_concentration, magnesium_concentration, temperature):
    # Calculate the folding energy of the sequence
    structure = RNA.fold(sequence)[0]
    dG = RNA.energy_of_struct(sequence, structure)

    print(sodium_concentration, magnesium_concentration)
    # Adjust free energy for salt concentration
    dG_sodium = -0.5 * math.log10(sodium_concentration)  # Effect of Na+ concentration
    dG_magnesium = -1.0 * math.log10(magnesium_concentration)  # Effect of Mg2+ concentration

    # Calculate total free energy with salt adjustments
    total_dG = dG + dG_sodium + dG_magnesium

    return structure, total_dG

def binding_free_energy(primer, target, sodium_concentration, magnesium_concentration, temperature):
    struct, dG =  RNA.cofold(primer + "&" + target)
    # Adjust free energy for salt concentration
    dG_sodium = -0.5 * math.log10(sodium_concentration)  # Effect of Na+ concentration
    dG_magnesium = -1.0 * math.log10(magnesium_concentration)  # Effect of Mg2+ concentration
    # Calculate total free energy with salt adjustments
    total_dG = dG + dG_sodium + dG_magnesium

    print(struct)
    return struct, total_dG

def calculate_percentages(hairpin_primer, hairpin_target, binding, temperature):
    # Calculate equilibrium constants based on free energies
    K_hairpin_primer = math.exp(-hairpin_primer / (R * (temperature + 273.15)))  # Convert temperature to Kelvin
    K_hairpin_target = math.exp(-hairpin_target / (R * (temperature + 273.15)))
    K_binding = math.exp(-binding / (R * (temperature + 273.15)))

    print(f"hp_primer_dg: {hairpin_primer}, hp_target_dg: {hairpin_target}, binding_dg: {binding}")

    # Debug prints to check values
    print(f"K_hairpin_primer: {K_hairpin_primer}, K_hairpin_target: {K_hairpin_target}, K_binding: {K_binding}")

    total = K_hairpin_primer + K_hairpin_target + K_binding

    # Ensure total is not zero to avoid division by zero error
    if total == 0:
        return 0, 0, 0, 100  # Assign everything to random coil if total is zero

    hairpin_primer_percent = (K_hairpin_primer / (K_hairpin_primer + K_binding)) * 100
    hairpin_target_percent = (K_hairpin_target / (K_hairpin_target + K_binding)) * 100
    binding_percent = (K_binding / total) * 100
    random_coil_percent = 100 - (hairpin_primer_percent + hairpin_target_percent + binding_percent)

    # More debug prints to check calculated percentages
    print(f"Hairpin Primer Percent: {hairpin_primer_percent}, Hairpin Target Percent: {hairpin_target_percent}, "
          f"Binding Percent: {binding_percent}, Random Coil Percent: {random_coil_percent}")

    return hairpin_primer_percent, hairpin_target_percent, binding_percent, random_coil_percent


@app.route('/api/simulate_binding', methods=['POST'])
def simulate_binding():
    data = request.json
    primer_sequence = data.get('primer')
    target_sequence = data.get('target')
    sodium_concentration = float(data.get('sodium_concentration', 0.0))
    magnesium_concentration = float(data.get('magnesium_concentration', 0.0))
    oligo_concentration = data.get('oligo_concentration', 1e-6)
    target_concentration = data.get('target_concentration', 1e-6)
    temperature = int(data.get('temperature', 25))

    hairpin_primer_struct, hairpin_primer_energy = calculate_free_energy(primer_sequence, sodium_concentration, magnesium_concentration, temperature)
    hairpin_target_struct, hairpin_target_energy = calculate_free_energy(target_sequence, sodium_concentration, magnesium_concentration, temperature)
    binding_struct, binding_energy = binding_free_energy(primer_sequence, target_sequence, sodium_concentration, magnesium_concentration, temperature)

    hairpin_primer_percent, hairpin_target_percent, binding_percent, random_coil_percent = calculate_percentages(
        hairpin_primer_energy, hairpin_target_energy, binding_energy, temperature
    )

    return jsonify({
        'hairpin_primer_energy': hairpin_primer_energy,
        'hairpin_target_energy': hairpin_target_energy,
        'binding_energy': binding_energy,
        'hairpin_primer_percent': hairpin_primer_percent,
        'hairpin_target_percent': hairpin_target_percent,
        'binding_percent': binding_percent,
        'random_coil_percent': random_coil_percent,
        'hairpin_primer_struct': hairpin_primer_struct,
        'hairpin_target_struct': hairpin_target_struct,
        'binding_struct': binding_struct,
    })



#####################
# GPT
#####################

## gpt
# import openai

# @app.route('/gpt', methods=['POST'])
# def gpt():
#     user_input = request.form.get('prompt')

#     try:
#         # Use the new `openai.completions` endpoint
#         response = openai.completions.create(
#             model="gpt-3.5-turbo",  # Use your preferred GPT model
#             messages=[{"role": "user", "content": user_input}],
#             max_tokens=100
#         )
#         generated_text = response.choices[0].message.content.strip()
#         return jsonify({'response': generated_text})
#     except Exception as e:
#         return jsonify({'error': str(e)})

######################
# stocks
######################

import yfinance as yf
import pandas as pd

# Function to generate buy/sell signals
def generate_signals(ticker):
    # Fetch stock data for the past year
    stock_data = yf.download(ticker, period="1y")
    
    # Calculate 50-day and 200-day Simple Moving Averages (SMA)
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    
    # Create signals based on SMA crossover
    stock_data['Signal'] = 0  # Initialize signal column
    stock_data['Signal'][50:] = [
        1 if stock_data['SMA_50'][i] > stock_data['SMA_200'][i] else -1
        for i in range(50, len(stock_data))
    ]
    
    # Create a Buy/Sell column based on Signal changes
    stock_data['Position'] = stock_data['Signal'].diff()

    # Return buy/sell signals (Position = 1 is Buy, Position = -1 is Sell)
    buy_signals = stock_data[stock_data['Position'] == 1]
    sell_signals = stock_data[stock_data['Position'] == -1]

    return buy_signals, sell_signals

# Function to convert Pandas Timestamps to strings in the DataFrame before returning as JSON
def format_signals(signals):
    signals.index = signals.index.strftime('%Y-%m-%d')  # Format timestamps as strings
    return signals[['Close', 'SMA_50', 'SMA_200']].tail(5).to_dict()

@app.route('/signals')
def get_signals():
    tickers = request.args.get('tickers')
    if not tickers:
        return jsonify({"error": "No tickers provided."}), 400

    ticker_list = tickers.split(',')
    results = {}

    for ticker in ticker_list:
        logging.info(f'Processing ticker: {ticker}')  # Log the ticker being processed
        try:
            data = yf.download(ticker, period='1mo', interval='1d')
            if data.empty:
                results[ticker] = {"error": "No data found."}
                continue
            
            # Add your signal generation logic here...
                        # Calculate moving averages
            data['SMA_5'] = data['Close'].rolling(window=5).mean()  # Short-term SMA
            data['SMA_20'] = data['Close'].rolling(window=20).mean()  # Long-term SMA

            # Initialize lists to store signals
            buy_signals = []
            sell_signals = []

            # Generate signals
            for i in range(1, len(data)):
                # Check for buy signal
                if data['SMA_5'].iloc[i] > data['SMA_20'].iloc[i] and data['SMA_5'].iloc[i - 1] <= data['SMA_20'].iloc[i - 1]:
                    buy_signals.append(data.index[i].strftime('%Y-%m-%d'))  # Add buy signal date
                # Check for sell signal
                elif data['SMA_5'].iloc[i] < data['SMA_20'].iloc[i] and data['SMA_5'].iloc[i - 1] >= data['SMA_20'].iloc[i - 1]:
                    sell_signals.append(data.index[i].strftime('%Y-%m-%d'))  # Add sell signal date
            # end signal logic
            
            results[ticker] = {
                "buy_signals": list(buy_signals),  # Convert set to list
                "sell_signals": list(sell_signals)  # Convert set to list
            }

        except Exception as e:
            logging.error(f'Error processing ticker {ticker}: {e}')
            results[ticker] = {"error": str(e)}

    return jsonify(results)


##########################
# image processing
#######################
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageFont, ImageDraw

@app.route('/process_image', methods=['POST'])
def process_image():
    image_file = request.files['image']
    mode = request.form.get('mode')
    factor = int(request.form.get('factor', 5))  # For modes needing an intensity

    # Load image
    img = Image.open(image_file)

    # Apply the selected mode
    if mode == 'grayscale':
        img = img.convert('L')
    elif mode == 'blur':
        img = img.filter(ImageFilter.GaussianBlur(factor))
    elif mode == 'edge_detection':
        img = img.filter(ImageFilter.FIND_EDGES)
    elif mode == 'sepia':
        img = apply_sepia(img)
    elif mode == 'pixelate':
        img = pixelate_image(img, factor)
    elif mode == 'invert':
        img = ImageOps.invert(img)
    elif mode == 'watermark':
        img = add_watermark(img, 'Sample Watermark')
    elif mode == 'sharpen':
        img = img.filter(ImageFilter.SHARPEN)
    elif mode == 'thumbnail':
        img.thumbnail((150, 150))
    elif mode == 'rotate':
        img = img.rotate(factor)

    # Save processed image to a bytes buffer
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/png')

def apply_sepia(img):
    width, height = img.size
    pixels = img.load()
    for py in range(height):
        for px in range(width):
            r, g, b = img.getpixel((px, py))
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
            pixels[px, py] = (min(255, tr), min(255, tg), min(255, tb))
    return img

def pixelate_image(img, pixel_size=9):
    img = img.resize((img.size[0] // pixel_size, img.size[1] // pixel_size), Image.NEAREST)
    img = img.resize((img.size[0] * pixel_size, img.size[1] * pixel_size), Image.NEAREST)
    return img

def add_watermark(img, text):
    img = img.convert('RGBA')
    txt = Image.new('RGBA', img.size, (255, 255, 255, 0))
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(txt)
    text_width, text_height = draw.textsize(text, font=font)
    position = (img.width - text_width - 10, img.height - text_height - 10)
    draw.text(position, text, fill=(255, 255, 255, 128), font=font)
    watermarked = Image.alpha_composite(img, txt)
    return watermarked.convert('RGB')



#######
# word  cloud
#######
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Generate the word cloud
    current_directory = os.getcwd()
    font_path = current_directory+"/static/fonts/Montserrat-Black.ttf"
    # font path sometimes cause issues with pythonanywhere. so disable this unless necessary.
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis',
                          ).generate(text)

    # Save the word cloud to a BytesIO object
    img_io = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_io, format='PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from wordcloud import WordCloud
import io
from flask import Flask, jsonify, send_file, request
import matplotlib.pyplot as plt


def create_mask(word, font_path, width=800, height=400, font_size=800):  # Default size matching WordCloud
    # Create a blank image with a black background
    mask_image = Image.new('L', (width, height), 0)  # 'L' mode for grayscale
    draw = ImageDraw.Draw(mask_image)

    # Load font
    try:
        font = ImageFont.truetype(font_path, size=font_size)  # Use the provided font size
    except IOError:
        font = ImageFont.load_default()

    # Calculate text bounding box
    text_bbox = draw.textbbox((0, 0), word, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate position to center the text
    text_x = (width - text_width) // 2
    # text_y = (height - text_height) // 4  # Center vertically
    text_y = font_size/20

    print(f"Text Bounding Box: {text_bbox}")
    print(f"Text Width: {text_width}, Text Height: {text_height}")
    print(f"Text X: {text_x}, Text Y: {text_y}")

    # Draw the word onto the mask image
    draw.text((text_x, text_y), word, fill=255, font=font)  # Fill white for the mask

    return np.array(mask_image)

def random_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    import random
    return f'#{random.randint(0, 0xFFFFFF):06x}'

@app.route('/generate_wordcloud_advanced', methods=['POST'])
def generate_wordcloud_advanced():
    data = request.get_json()
    text = data.get('text', '')
    word = data.get('word', '')            # Get the word for mask
    font_size = int(data.get('font_size', 800))  # Get the font size for mask

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    current_directory = os.getcwd()
    font_path = os.path.join(current_directory, "static/fonts/Montserrat-Black.ttf")

    canvas_width = 800
    canvas_height = 400

    mask = None
    if word:
        mask = create_mask(word, font_path, width=canvas_width, height=canvas_height, font_size=font_size)

    wordcloud = WordCloud(
        width=canvas_width,
        height=canvas_height,
        background_color='black',
        colormap='viridis',
        mask=mask,
        color_func=random_color_func
    ).generate(text)

    img_io = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_io, format='PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


#################
# Function to generate proper spiral coordinates
def generate_spiral(words):
    angle_increment = 0.2  # Angle increment for spiral
    radius_increment = 5   # Radius increment per word
    spiral_data = []
    angle = 0

    for index, word in enumerate(words):
        radius = radius_increment * index  # Increase radius for each word
        x = radius * math.cos(angle)  # Calculate x position
        y = radius * math.sin(angle)  # Calculate y position
        
        angle += angle_increment  # Increment the angle for the next word

        # Set word size and color
        size = 20 + (index % 15)  # Vary the size between 20 and 35
        color = f"hsl({(index * 30) % 360}, 70%, 60%)"  # Different color for each word

        spiral_data.append({
            "word": word,
            "x": x,
            "y": y,
            "size": size,
            "color": color
        })
    
    return spiral_data

@app.route('/generate-spiral', methods=['POST'])
def generate_spiral_endpoint():
    data = request.json
    words = data.get('words', [])

    if not words:
        return jsonify({'error': 'No words provided'}), 400

    spiral_data = generate_spiral(words)
    return jsonify(spiral_data)

#############
# bubble chart
#############
from collections import Counter

@app.route('/generate-bubbles', methods=['POST'])
def generate_bubbles():
    data = request.json
    words = data.get('words', [])

    if not words:
        return jsonify({'error': 'No words provided'}), 400

    # Count word frequencies
    word_counts = Counter(words)
    bubble_data = [{'word': word, 'size': count * 10} for word, count in word_counts.items()]  # Adjust size factor as needed
    return jsonify(bubble_data)

messages = []
@app.route('/get_messages', methods=['GET'])
def get_messages():
    return jsonify(messages)

@app.route('/add_message', methods=['POST'])
def add_message():
    global messages
    data = request.json
    if 'message' in data:
        messages.append(data['message'])
        # Limit the number of messages to 280 (20 columns × 14 rows)
        if len(messages) > 280:
            messages = messages[-280:]
        return jsonify({"status": "success", "message": "Message added"})
    return jsonify({"status": "error", "message": "Invalid request"}), 400


if __name__ == "__main__":
    app.run(debug=True)

