import requests
import json
import time
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pytz
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(_BASE_DIR, 'config.json')

API_BASE_URL = "https://api.oligodesign.com/api/stocks"
SIGNALS_API_URL = f"{API_BASE_URL}/signals"
HISTORY_FILE = os.path.join(_BASE_DIR, 'signal_history.json')

POLL_INTERVAL_SECONDS = int(os.getenv("AGENT_POLL_SECONDS", "60"))
# BATCH_SIZE and BATCH_PAUSE_SECONDS are read from config.json at runtime

ET_TZ = pytz.timezone('US/Eastern')


# ---------------------------------------------------------------------------
# Config helpers — live-read every run so admin changes take effect immediately
# ---------------------------------------------------------------------------

def _load_config():
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    cfg.setdefault('tickers', ['AAPL', 'GOOGL', 'MSFT'])
    cfg.setdefault('run_times', ['09:35', '15:55'])
    cfg.setdefault('to_email', os.getenv('TO_EMAIL', 'linlifeng@gmail.com'))
    cfg.setdefault('batch_size', int(os.getenv('BATCH_SIZE', '20')))
    cfg.setdefault('batch_pause_seconds', int(os.getenv('BATCH_PAUSE_SECONDS', '300')))
    return cfg


def _now_et():
    return datetime.now(ET_TZ)


def _is_weekday_et():
    return _now_et().weekday() < 5


def _today_et():
    return _now_et().strftime('%Y-%m-%d')


def _hhmm_et():
    return _now_et().strftime('%H:%M')

def load_history():
    """Load signal history from file"""
    try:
        with open(HISTORY_FILE) as f:
            history = json.load(f)
        if not isinstance(history, dict):
            return {}
        history.setdefault("__meta__", {})
        history["__meta__"].setdefault("sent_slots", {})
        return history
    except FileNotFoundError:
        print("No history file found. Starting fresh.")
        return {"__meta__": {"sent_slots": {}}}
    except json.JSONDecodeError as e:
        print(f"Error reading history file: {e}. Starting fresh.")
        return {"__meta__": {"sent_slots": {}}}

def save_history(history):
    """Save signal history to file"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

def send_email_notification(subject, body, to_email=None):
    """Send email notification via Zoho-compatible SMTP."""
    try:
        smtp_host = os.getenv("SMTP_HOST", "smtp.zoho.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USERNAME") or os.getenv("EMAIL_ADDRESS")
        smtp_password = os.getenv("SMTP_PASSWORD") or os.getenv("EMAIL_PASSWORD")
        smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() not in {"0", "false", "no"}
        mail_from = os.getenv("MAIL_FROM", smtp_username or "")
        if not to_email:
            to_email = os.getenv("TO_EMAIL", "linlifeng@gmail.com")

        if not smtp_username or not smtp_password:
            print("SMTP credentials missing (SMTP_USERNAME/SMTP_PASSWORD or EMAIL_ADDRESS/EMAIL_PASSWORD)")
            return False

        recipients = [
            r.strip() for r in str(to_email).replace(';', ',').split(',') if r.strip()
        ]
        if not recipients:
            print("No recipient email configured")
            return False

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = mail_from
        msg['To'] = ', '.join(recipients)
        # Set plain-text content first, then attach HTML alternative
        import re as _re
        plain_fallback = _re.sub(r'<[^>]+>', ' ', body).strip() if '<html' in body else body
        msg.set_content(plain_fallback)
        if '<html' in body:
            msg.add_alternative(body, subtype='html')

        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
            if smtp_use_tls:
                smtp.starttls()
            smtp.login(smtp_username, smtp_password)
            smtp.send_message(msg, to_addrs=recipients)
        
        print("Email notification sent successfully")
        return True
        
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def _fetch_signals_batched(tickers, batch_size=20, batch_pause=300):
    """Fetch signals in batches of BATCH_SIZE with a pause between batches."""
    results = {}
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    total = len(batches)

    for idx, batch in enumerate(batches, 1):
        if total > 1:
            print(f"  Batch {idx}/{total}: {', '.join(batch)}")
        try:
            response = requests.get(
                SIGNALS_API_URL,
                params={'tickers': ','.join(batch)},
                timeout=120,
            )
            response.raise_for_status()
            results.update(response.json())
        except Exception as e:
            print(f"  Batch {idx} error: {e}")
            for t in batch:
                results[t] = {'error': str(e)}

        if idx < total:
            print(f"  Pausing {batch_pause}s before next batch…")
            time.sleep(batch_pause)

    return results


# ---------------------------------------------------------------------------
# HTML email helpers
# ---------------------------------------------------------------------------

_SIG_COLORS = {
    'ENTER':  {'bg': '#dcfce7', 'border': '#16a34a', 'badge_bg': '#16a34a', 'text': '#14532d'},
    'WATCH':  {'bg': '#fefce8', 'border': '#ca8a04', 'badge_bg': '#ca8a04', 'text': '#713f12'},
    'IGNORE': {'bg': '#f8fafc', 'border': '#cbd5e1', 'badge_bg': '#94a3b8', 'text': '#475569'},
}


def _sig_badge(sig):
    c = _SIG_COLORS.get(sig, _SIG_COLORS['IGNORE'])
    return (
        f'<span style="background:{c["badge_bg"]};color:#fff;'
        f'padding:2px 10px;border-radius:12px;font-size:12px;'
        f'font-weight:700;letter-spacing:.5px;">{sig}</span>'
    )


def _strength_bar_html(strength):
    filled = min(5, max(0, strength))
    bars = ''.join(
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
        f'margin:0 1px;background:{"#16a34a" if i < filled else "#e2e8f0"};"></span>'
        for i in range(5)
    )
    return f'<span style="vertical-align:middle">{bars}</span>'


def _ticker_row_html(ticker, data):
    sig      = data.get('signal', 'IGNORE')
    conf     = data.get('confidence', 0)
    setup    = data.get('setupType', 'NONE')
    regime   = data.get('marketRegime', '?')
    entry    = data.get('entryPrice', 0)
    stop     = data.get('stopLoss', 0)
    target   = data.get('priceTarget', 0)
    rr       = data.get('riskReward', '0.00')
    xret     = data.get('expectedReturn', '0.0')
    reasons  = data.get('reasons', [])
    strength = data.get('strength', 0)
    c        = _SIG_COLORS.get(sig, _SIG_COLORS['IGNORE'])

    detail = ''
    if sig in ('ENTER', 'WATCH'):
        detail = (
            f'<tr style="background:{c["bg"]};">'
            f'<td colspan="2" style="padding:4px 16px 8px 56px;color:#374151;font-size:13px;">'
            f'Entry <b>${entry}</b> &nbsp;·&nbsp; '
            f'Stop <b>${stop}</b> &nbsp;·&nbsp; '
            f'Target <b>${target}</b> &nbsp;·&nbsp; '
            f'R/R <b>{rr}</b> &nbsp;·&nbsp; '
            f'Exp <b>+{xret}%</b>'
            f'</td></tr>'
        )
    reasons_html = ''
    if reasons:
        reasons_html = (
            f'<tr style="background:{c["bg"]};">'
            f'<td colspan="2" style="padding:2px 16px 10px 56px;color:#6b7280;font-size:12px;">'
            + ' &nbsp;|&nbsp; '.join(reasons) +
            f'</td></tr>'
        )

    return (
        f'<tr style="background:{c["bg"]};border-left:4px solid {c["border"]};'
        f'border-bottom:1px solid #e2e8f0;">'
        f'<td style="padding:10px 12px 10px 16px;font-weight:700;font-size:15px;'
        f'color:#111;width:70px">{ticker}</td>'
        f'<td style="padding:10px 12px;">'
        f'{_sig_badge(sig)} &nbsp; '
        f'{_strength_bar_html(strength)} &nbsp; '
        f'<span style="font-size:12px;color:#6b7280;">conf:{conf}% &nbsp; {setup} &nbsp; {regime}</span>'
        f'</td></tr>'
        + detail + reasons_html
    )


def _build_html_email(enter_data, watch_data, ignore_data, errors, tickers, now_str):
    """Build a full HTML email string."""
    def section(title, color, rows_html):
        if not rows_html:
            return ''
        return (
            f'<tr><td colspan="2" style="padding:18px 16px 6px;'
            f'font-size:13px;font-weight:700;color:{color};'
            f'letter-spacing:.5px;border-top:2px solid {color};">{title}</td></tr>'
            + rows_html
        )

    enter_rows  = ''.join(_ticker_row_html(t, d) for t, d in enter_data)
    watch_rows  = ''.join(_ticker_row_html(t, d) for t, d in watch_data)
    ignore_rows = ''.join(_ticker_row_html(t, d) for t, d in ignore_data)
    error_rows  = ''.join(
        f'<tr><td colspan="2" style="padding:6px 16px;color:#dc2626;font-size:13px;">{e}</td></tr>'
        for e in errors
    )

    counts = (
        f'<b style="color:#16a34a">{len(enter_data)} ENTER</b> &nbsp; '
        f'<b style="color:#ca8a04">{len(watch_data)} WATCH</b> &nbsp; '
        f'<b style="color:#94a3b8">{len(ignore_data)} IGNORE</b>'
    )

    body_rows = (
        section('▲ ENTER', '#16a34a', enter_rows)
        + section('◎ WATCH', '#ca8a04', watch_rows)
        + section('— IGNORE', '#94a3b8', ignore_rows)
        + (section('⚠ ERRORS', '#dc2626', error_rows) if error_rows else '')
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:32px 0;">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="background:#fff;border-radius:12px;
  box-shadow:0 2px 8px rgba(0,0,0,.08);overflow:hidden;">

  <!-- header -->
  <tr style="background:#0f172a;">
    <td colspan="2" style="padding:20px 24px;">
      <span style="font-size:20px;font-weight:700;color:#fff;">📈 Stock Signal Digest</span><br>
      <span style="font-size:12px;color:#94a3b8;">{now_str} &nbsp;·&nbsp; {len(tickers)} tickers scanned &nbsp;·&nbsp; {counts}</span>
    </td>
  </tr>

  <!-- rows -->
  {body_rows}

  <!-- footer -->
  <tr><td colspan="2" style="padding:16px 24px;background:#f8fafc;
    border-top:1px solid #e2e8f0;font-size:11px;color:#94a3b8;">
    Stock Signal Agent &nbsp;·&nbsp; oligodesign.com
  </td></tr>

</table>
</td></tr></table>
</body></html>"""


def _format_signal_line(ticker, data):
    """Plain-text fallback line (kept for console output)."""
    sig      = data.get('signal', 'IGNORE')
    conf     = data.get('confidence', 0)
    setup    = data.get('setupType', 'NONE')
    regime   = data.get('marketRegime', '?')
    entry    = data.get('entryPrice', 0)
    stop     = data.get('stopLoss', 0)
    target   = data.get('priceTarget', 0)
    rr       = data.get('riskReward', '0.00')
    xret     = data.get('expectedReturn', '0.0')
    reasons  = data.get('reasons', [])
    strength = data.get('strength', 0)
    bars     = '█' * strength + '░' * (5 - strength)

    line = (
        f"  {ticker:<6}  {sig:<6}  conf:{conf:>3}%  [{bars}]  "
        f"setup:{setup}  regime:{regime}\n"
    )
    if sig in ('ENTER', 'WATCH'):
        line += (
            f"          entry:${entry}  stop:${stop}  target:${target}  "
            f"R/R:{rr}  exp:+{xret}%\n"
        )
    if reasons:
        line += f"          {' | '.join(reasons)}\n"
    return line, sig


def check_signals():
    """Fetch signals for all tickers in batches, then email a rich HTML digest."""
    cfg = _load_config()
    tickers  = cfg['tickers']
    to_email = cfg['to_email']

    if not tickers:
        print("No tickers in config. Skipping.")
        return

    print(f"Scanning {len(tickers)} tickers in batches of {cfg['batch_size']}…")
    result = _fetch_signals_batched(tickers, cfg['batch_size'], cfg['batch_pause_seconds'])

    enter_data, watch_data, ignore_data, error_lines = [], [], [], []
    plain_lines = []

    for ticker in tickers:
        data = result.get(ticker, {'error': 'no response'})
        if 'error' in data:
            error_lines.append(f"{ticker}: {data['error']}")
            plain_lines.append(f"  {ticker}: ERROR — {data['error']}\n")
            continue
        line, sig = _format_signal_line(ticker, data)
        plain_lines.append(line)
        if sig == 'ENTER':
            enter_data.append((ticker, data))
        elif sig == 'WATCH':
            watch_data.append((ticker, data))
        else:
            ignore_data.append((ticker, data))

    now_str = _now_et().strftime('%Y-%m-%d %H:%M ET')

    # Plain-text version (console + fallback)
    plain  = f"Stock scan — {now_str}\n"
    plain += f"Tickers: {len(tickers)}  ENTER:{len(enter_data)}  WATCH:{len(watch_data)}  IGNORE:{len(ignore_data)}\n"
    plain += "=" * 60 + "\n"
    plain += "".join(plain_lines)
    print(plain)

    # HTML version
    html_body = _build_html_email(enter_data, watch_data, ignore_data, error_lines, tickers, now_str)

    if enter_data:
        subject = f"[Stock] {len(enter_data)} ENTER signal(s) — {now_str}"
    elif watch_data:
        subject = f"[Stock] {len(watch_data)} WATCH — {now_str}"
    else:
        subject = f"[Stock] Digest (no signals) — {now_str}"

    send_email_notification(subject, html_body, to_email)

    history = load_history()
    save_history(history)

def _slot_already_sent_today(history, slot_hhmm):
    sent_slots = history.get("__meta__", {}).get("sent_slots", {})
    return slot_hhmm in sent_slots.get(_today_et(), [])


def _mark_slot_sent_today(history, slot_hhmm):
    history.setdefault("__meta__", {}).setdefault("sent_slots", {})
    today_slots = history["__meta__"]["sent_slots"].setdefault(_today_et(), [])
    if slot_hhmm not in today_slots:
        today_slots.append(slot_hhmm)


def run_agent():
    """Main agent loop: send exactly at configured ET slots on weekdays."""
    print("Stock Signal Agent starting...")
    print(f"Poll interval: {POLL_INTERVAL_SECONDS}s  (batch settings read from config.json)")

    while True:
        cfg = _load_config()  # re-read every tick so admin changes are live
        run_times = cfg.get('run_times', ['09:35', '15:55'])

        now = _now_et()
        now_hhmm = now.strftime('%H:%M')
        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S ET')}] heartbeat | slots: {', '.join(run_times)}")

        if _is_weekday_et() and now_hhmm in run_times:
            history = load_history()
            if _slot_already_sent_today(history, now_hhmm):
                print(f"⏭️  Slot {now_hhmm} already sent today; skipping.")
            else:
                print(f"✅ Slot {now_hhmm} ET — starting scan…")
                try:
                    check_signals()
                    history = load_history()
                    _mark_slot_sent_today(history, now_hhmm)
                    save_history(history)
                except Exception as e:
                    print(f"❌ Error during scheduled run: {e}")
        else:
            print("⏰ Not a scheduled send slot.")

        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    # Verify environment variables
    if not (
        (os.getenv("SMTP_USERNAME") and os.getenv("SMTP_PASSWORD"))
        or (os.getenv("EMAIL_ADDRESS") and os.getenv("EMAIL_PASSWORD"))
    ):
        print("⚠️  SMTP credentials are not set")
        print("Set SMTP_USERNAME/SMTP_PASSWORD (recommended for Zoho) or EMAIL_ADDRESS/EMAIL_PASSWORD")
    
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped by user")
    except Exception as e:
        print(f"💥 Agent crashed with error: {e}")