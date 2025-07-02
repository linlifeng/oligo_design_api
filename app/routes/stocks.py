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

