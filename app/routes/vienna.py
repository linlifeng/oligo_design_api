# app/routes/vienna.py
from flask import Blueprint, request, jsonify
import subprocess
import RNA

vienna_bp = Blueprint('vienna', __name__)

@vienna_bp.route('/simulate_binding', methods=['POST'])
def simulate_binding():
    data = request.json
    seq1 = data.get('seq1')
    seq2 = data.get('seq2')

    if not seq1 or not seq2:
        return jsonify({'error': 'Missing sequences'}), 400

    input_str = f"{seq1}&{seq2}\\n"
    try:
        result = subprocess.run(
            ['RNAcofold'],
            input=input_str.encode(),
            capture_output=True,
            check=True
        )
        output = result.stdout.decode()
        return jsonify({'result': output})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': e.stderr.decode()}), 500



@vienna_bp.route('/calculate_hairpin', methods=['POST'])
def calculate_hairpin(n=20):
    data = request.json
    sequence = data.get('sequence')

    if not sequence:
        return jsonify({'error': 'Missing sequence'}), 400

    fc = RNA.fold_compound(sequence)
    mfe_structure, mfe = fc.mfe()

    # Get multiple suboptimal structures
    subopts = fc.subopt(n)  # get top n structures
    structures = []

    for sol in subopts:
        structures.append({
            'dot_bracket': sol.structure,
            'deltaG': round(sol.energy, 2)
        })

    return jsonify({
        'sequence': sequence,
        'structure_count': len(structures),
        'structures': structures
    })
