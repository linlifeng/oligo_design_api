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

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = mail_from
        msg['To'] = to_email
        msg.set_content(body)

        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
            if smtp_use_tls:
                smtp.starttls()
            smtp.login(smtp_username, smtp_password)
            smtp.send_message(msg)
        
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


def _format_signal_line(ticker, data):
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
    """Fetch signals for all tickers in batches, then email a rich digest."""
    cfg = _load_config()
    tickers  = cfg['tickers']
    to_email = cfg['to_email']

    if not tickers:
        print("No tickers in config. Skipping.")
        return

    print(f"Scanning {len(tickers)} tickers in batches of {cfg['batch_size']}…")
    result = _fetch_signals_batched(tickers, cfg['batch_size'], cfg['batch_pause_seconds'])

    enter_lines, watch_lines, ignore_lines, error_lines = [], [], [], []

    for ticker in tickers:
        data = result.get(ticker, {'error': 'no response'})
        if 'error' in data:
            error_lines.append(f"  {ticker}: {data['error']}")
            continue
        line, sig = _format_signal_line(ticker, data)
        if sig == 'ENTER':
            enter_lines.append(line)
        elif sig == 'WATCH':
            watch_lines.append(line)
        else:
            ignore_lines.append(line)

    now_str = _now_et().strftime('%Y-%m-%d %H:%M ET')
    body  = f"Stock scan — {now_str}\n"
    body += f"Tickers: {len(tickers)}  |  ENTER: {len(enter_lines)}  WATCH: {len(watch_lines)}  IGNORE: {len(ignore_lines)}\n"
    body += "=" * 60 + "\n\n"

    if enter_lines:
        body += f"── ENTER ({len(enter_lines)}) ──────────────────────────────\n"
        body += "".join(enter_lines) + "\n"

    if watch_lines:
        body += f"── WATCH ({len(watch_lines)}) ──────────────────────────────\n"
        body += "".join(watch_lines) + "\n"

    if ignore_lines:
        body += f"── IGNORE ({len(ignore_lines)}) ─────────────────────────────\n"
        body += "".join(ignore_lines) + "\n"

    if error_lines:
        body += f"── ERRORS ({len(error_lines)}) ─────────────────────────────\n"
        body += "\n".join(error_lines) + "\n"

    if enter_lines:
        subject = f"[Stock] {len(enter_lines)} ENTER signal(s) — {now_str}"
    elif watch_lines:
        subject = f"[Stock] {len(watch_lines)} WATCH — {now_str}"
    else:
        subject = f"[Stock] Digest (no signals) — {now_str}"

    print(body)
    send_email_notification(subject, body, to_email)

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
    print(f"Poll interval: {POLL_INTERVAL_SECONDS}s  Batch size: {BATCH_SIZE}  Batch pause: {BATCH_PAUSE_SECONDS}s")

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