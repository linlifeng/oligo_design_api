import requests
import json
import time
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pytz
import os

# Configuration
API_BASE_URL = "https://api.oligodesign.com/api/stocks"
TICKERS_API_URL = f"{API_BASE_URL}/tickers"
SIGNALS_API_URL = f"{API_BASE_URL}/signals"
HISTORY_FILE = "signal_history.json"

# Poll every minute, but only send at configured ET times.
POLL_INTERVAL_SECONDS = int(os.getenv("AGENT_POLL_SECONDS", "60"))
RUN_TIMES_ET = [
    t.strip() for t in os.getenv("RUN_TIMES_ET", "09:35,15:55").split(",")
    if t.strip()
]

ET_TZ = pytz.timezone('US/Eastern')


def _now_et():
    return datetime.now(ET_TZ)


def _is_weekday_et():
    return _now_et().weekday() < 5


def _today_et():
    return _now_et().strftime('%Y-%m-%d')


def _hhmm_et():
    return _now_et().strftime('%H:%M')

def load_tickers():
    """Load tickers from the API"""
    try:
        response = requests.get(TICKERS_API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('tickers', [])
    except requests.exceptions.RequestException as e:
        print(f"Error loading tickers from API: {e}")
        # Fallback to local file if API fails
        try:
            with open('tickers.json') as f:
                return json.load(f)['tickers']
        except FileNotFoundError:
            print("No local tickers file found. Using default tickers.")
            return ['AAPL', 'GOOGL', 'MSFT']  # Default tickers

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

def send_email_notification(subject, body):
    """Send email notification via Zoho-compatible SMTP."""
    try:
        smtp_host = os.getenv("SMTP_HOST", "smtp.zoho.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USERNAME") or os.getenv("EMAIL_ADDRESS")
        smtp_password = os.getenv("SMTP_PASSWORD") or os.getenv("EMAIL_PASSWORD")
        smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() not in {"0", "false", "no"}
        mail_from = os.getenv("MAIL_FROM", smtp_username or "")
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

def check_signals():
    """Check signals using the full signal engine and send a digest email."""
    print("Loading tickers...")
    tickers = load_tickers()

    if not tickers:
        print("No tickers found. Skipping signal check.")
        return

    print(f"Checking signals for {len(tickers)} tickers: {', '.join(tickers)}")

    ticker_param = ','.join(tickers)

    try:
        response = requests.get(SIGNALS_API_URL, params={'tickers': ticker_param}, timeout=60)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching signals from API: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing API response: {e}")
        return

    enter_signals = []
    watch_signals = []
    errors = []

    for ticker, data in result.items():
        if 'error' in data:
            errors.append(f"{ticker}: {data['error']}")
            continue

        sig = data.get('signal', 'IGNORE')
        conf = data.get('confidence', 0)
        reasons = data.get('reasons', [])
        entry = data.get('entryPrice', 0)
        stop = data.get('stopLoss', 0)
        target = data.get('priceTarget', 0)
        rr = data.get('riskReward', '0.00')
        regime = data.get('marketRegime', '')
        setup = data.get('setupType', '')

        line = (
            f"  [{ticker}] {sig} | Confidence: {conf}% | Setup: {setup} | Regime: {regime}\n"
            f"    Entry: ${entry}  Stop: ${stop}  Target: ${target}  R/R: {rr}\n"
            f"    {'; '.join(reasons)}"
        )

        if sig == 'ENTER':
            enter_signals.append(line)
        elif sig == 'WATCH':
            watch_signals.append(line)

    now_str = _now_et().strftime('%Y-%m-%d %H:%M:%S ET')
    body = f"Stock scan completed at {now_str}.\n\n"

    if enter_signals:
        body += f"ENTER ({len(enter_signals)} ticker(s)):\n"
        body += '\n'.join(enter_signals) + '\n\n'

    if watch_signals:
        body += f"WATCH ({len(watch_signals)} ticker(s)):\n"
        body += '\n'.join(watch_signals) + '\n\n'

    if not enter_signals and not watch_signals:
        body += "No ENTER or WATCH signals. All tickers are IGNORE.\n\n"

    if errors:
        body += f"Errors ({len(errors)}): {'; '.join(errors)}\n\n"

    body += f"Checked tickers: {', '.join(tickers)}\n"

    if enter_signals:
        subject = f"Stock Signal: {len(enter_signals)} ENTER signal(s)"
    elif watch_signals:
        subject = f"Stock Signal: {len(watch_signals)} WATCH signal(s)"
    else:
        subject = "Stock Scan Digest (No Signals)"

    print(body)
    send_email_notification(subject, body)

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
    print(f"Run times (ET): {', '.join(RUN_TIMES_ET)}")
    print(f"Poll interval: {POLL_INTERVAL_SECONDS}s")

    while True:
        now = _now_et()
        now_hhmm = now.strftime('%H:%M')
        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S ET')}] Agent heartbeat")

        if _is_weekday_et() and now_hhmm in RUN_TIMES_ET:
            history = load_history()
            if _slot_already_sent_today(history, now_hhmm):
                print(f"⏭️ Slot {now_hhmm} ET already sent today; skipping.")
            else:
                print(f"✅ Scheduled slot {now_hhmm} ET reached - running scan + email...")
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