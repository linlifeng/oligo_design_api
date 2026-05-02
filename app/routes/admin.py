# app/routes/admin.py
import json
import os
import functools
from flask import Blueprint, request, Response, render_template

admin_bp = Blueprint('admin', __name__, template_folder='../templates')

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_FILE = os.path.join(_BASE_DIR, 'config.json')


def _load_config():
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        cfg.setdefault('to_email', 'linlifeng@gmail.com')
        cfg.setdefault('run_times', ['09:35', '15:55'])
        cfg.setdefault('tickers', [])
        cfg.setdefault('batch_size', 20)
        cfg.setdefault('batch_pause_seconds', 300)
        cfg.setdefault('strategy', {})
        s = cfg['strategy']
        s.setdefault('weights', {})
        s['weights'].setdefault('sma_align', 3.0)
        s['weights'].setdefault('price_mom', 2.0)
        s['weights'].setdefault('rsi', 2.0)
        s['weights'].setdefault('macd', 1.5)
        s.setdefault('thresholds', {})
        s['thresholds'].setdefault('watch_confidence_min', 40)
        s['thresholds'].setdefault('enter_confidence_min', 45)
        s['thresholds'].setdefault('rsi_oversold', 35)
        s['thresholds'].setdefault('rsi_overbought', 70)
        s['thresholds'].setdefault('cci_oversold', -100)
        s.setdefault('setups', {'TREND': True, 'MEAN_REVERSION': True, 'PULLBACK': True})
        return cfg
    except Exception:
        return {'to_email': '', 'run_times': ['09:35', '15:55'], 'tickers': [],
                'batch_size': 20, 'batch_pause_seconds': 300, 'strategy': {}}


def _save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)


def _require_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        admin_user = os.getenv('ADMIN_USER', '')
        admin_pass = os.getenv('ADMIN_PASSWORD', '')
        auth = request.authorization
        if not auth or auth.username != admin_user or auth.password != admin_pass:
            return Response(
                'Unauthorized', 401,
                {'WWW-Authenticate': 'Basic realm="Stock Admin"'}
            )
        return f(*args, **kwargs)
    return decorated


@admin_bp.route('/admin', methods=['GET', 'POST'])
@_require_auth
def admin():
    message = None
    cfg = _load_config()

    if request.method == 'POST':
        try:
            raw_tickers = request.form.get('tickers', '')
            tickers = [t.strip().upper() for t in raw_tickers.replace(',', '\n').splitlines() if t.strip()]
            tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

            run_time_1 = request.form.get('run_time_1', '09:35').strip()
            run_time_2 = request.form.get('run_time_2', '15:55').strip()
            to_email   = request.form.get('to_email', '').strip()

            batch_size  = max(1, int(request.form.get('batch_size', '20') or '20'))
            batch_pause = max(0, int(request.form.get('batch_pause', '300') or '300'))

            cfg['tickers']              = tickers
            cfg['run_times']            = [run_time_1, run_time_2]
            cfg['to_email']             = to_email
            cfg['batch_size']           = batch_size
            cfg['batch_pause_seconds']  = batch_pause

            # strategy
            def _fv(key, default): return float(request.form.get(key, default) or default)
            def _iv(key, default): return int(float(request.form.get(key, default) or default))
            cfg['strategy'] = {
                'weights': {
                    'sma_align': _fv('w_sma_align', 3.0),
                    'price_mom': _fv('w_price_mom', 2.0),
                    'rsi':       _fv('w_rsi',       2.0),
                    'macd':      _fv('w_macd',      1.5),
                },
                'thresholds': {
                    'watch_confidence_min': _iv('t_watch_confidence_min', 40),
                    'enter_confidence_min': _iv('t_enter_confidence_min', 45),
                    'rsi_oversold':         _iv('t_rsi_oversold',         35),
                    'rsi_overbought':       _iv('t_rsi_overbought',       70),
                    'cci_oversold':         _iv('t_cci_oversold',        -100),
                },
                'setups': {
                    'TREND':          request.form.get('s_TREND')          == 'on',
                    'MEAN_REVERSION': request.form.get('s_MEAN_REVERSION') == 'on',
                    'PULLBACK':       request.form.get('s_PULLBACK')       == 'on',
                },
            }

            _save_config(cfg)
            message = ('success', f'Saved — {len(tickers)} tickers, runs at {run_time_1} & {run_time_2} ET, alerts → {to_email}')
        except Exception as e:
            message = ('error', f'Save failed: {e}')

    return render_template('admin.html', cfg=cfg, message=message)
