# app/routes/stocks.py
from flask import Blueprint, request, jsonify
import logging
import yfinance as yf

stocks_bp = Blueprint('stocks', __name__)

@stocks_bp.route('/signals', methods=['GET'])
def get_signals():
    tickers = request.args.get('tickers')
    if not tickers:
        return jsonify({"error": "No tickers provided."}), 400

    ticker_list = tickers.split(',')
    results = {}

    for ticker in ticker_list:
        logging.info(f'Processing ticker: {ticker}')
        try:
            data = yf.download(ticker, period='1mo', interval='1d')
            if data.empty:
                results[ticker] = {"error": "No data found."}
                continue

            # Calculate moving averages
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()

            buy_signals = []
            sell_signals = []

            for i in range(1, len(data)):
                if data['SMA_5'].iloc[i] > data['SMA_20'].iloc[i] and data['SMA_5'].iloc[i - 1] <= data['SMA_20'].iloc[i - 1]:
                    buy_signals.append(data.index[i].strftime('%Y-%m-%d'))
                elif data['SMA_5'].iloc[i] < data['SMA_20'].iloc[i] and data['SMA_5'].iloc[i - 1] >= data['SMA_20'].iloc[i - 1]:
                    sell_signals.append(data.index[i].strftime('%Y-%m-%d'))

            results[ticker] = {
                "buy_signals": buy_signals,
                "sell_signals": sell_signals
            }

        except Exception as e:
            logging.error(f'Error processing ticker {ticker}: {e}')
            results[ticker] = {"error": str(e)}

    return jsonify(results)



@stocks_bp.route('/stock', methods=['GET'])
def get_stock_data():
    symbol = request.args.get('symbol', '').upper()

    if not symbol:
        return jsonify({'error': 'Missing stock symbol'}), 400

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='90d')  # Up to 60 days

        if hist.empty:
            return jsonify({'error': 'No data returned'}), 404

        hist = hist.sort_index(ascending=False)

        time_series = {}

        for date, row in hist.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            time_series[date_str] = {
                "1. open": f"{row['Open']:.4f}",
                "2. high": f"{row['High']:.4f}",
                "3. low": f"{row['Low']:.4f}",
                "4. close": f"{row['Close']:.4f}",
                "5. volume": str(int(row['Volume']))
            }

        last_refreshed = max(time_series.keys())

        response = {
            "Meta Data": {
                "1. Information": "Daily Prices (open, high, low, close) and Volumes",
                "2. Symbol": symbol,
                "3. Last Refreshed": last_refreshed,
                "4. Output Size": "Compact",
                "5. Time Zone": "US/Eastern"
            },
            "Time Series (Daily)": time_series
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

