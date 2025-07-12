# app/routes/oligo.py
from flask import Blueprint, request, jsonify
import primer3

oligo_bp = Blueprint('oligo', __name__)

@oligo_bp.route('/calculate_tm', methods=['POST'])
def calculate_tm():
    data = request.json
    sequence = data.get('sequence')
    if not sequence:
        return jsonify({'error': 'Missing sequence'}), 400

    tm = primer3.calcTm(sequence)
    return jsonify({'tm': tm})


@oligo_bp.route('/calculate_hairpin', methods=['POST'])
def calculate_hairpin():
    data = request.json
    sequence = data.get('sequence')
    if not sequence:
        return jsonify({'error': 'Missing sequence'}), 400

    hairpin = primer3.calc_hairpin(sequence)
    # print(hairpin)
    return jsonify({
        'structure_found': hairpin.structure_found,
        'deltaG': hairpin.dg,
        'temperature': hairpin.tm
    })


@oligo_bp.route('/design_oligo', methods=['POST'])
def design_oligo():
    data = request.json
    template = data.get('template')
    if not template:
        return jsonify({'error': 'Missing template'}), 400

    result = primer3.bindings.designPrimers({
        'SEQUENCE_ID': 'oligo',
        'SEQUENCE_TEMPLATE': template
    }, {
        'PRIMER_OPT_SIZE': 20,
        'PRIMER_MIN_SIZE': 18,
        'PRIMER_MAX_SIZE': 25,
        'PRIMER_NUM_RETURN': 1
    })

    return jsonify({
        'left_primer': result.get('PRIMER_LEFT_0_SEQUENCE'),
        'right_primer': result.get('PRIMER_RIGHT_0_SEQUENCE')
    })

