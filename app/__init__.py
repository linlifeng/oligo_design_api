from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

def create_app():
    load_dotenv()  # Load from .env file

    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret')

    CORS(app)

    # Register Blueprints
    from .routes.stocks import stocks_bp
    # from .routes.oligo import oligo_bp
    # from .routes.vienna import vienna_bp
    # from .routes.image import image_bp
    # from .routes.wordcloud import wordcloud_bp
    # from .routes.art import art_bp
    # from .routes.chat import chat_bp

    app.register_blueprint(stocks_bp, url_prefix='/api/stocks')
    # app.register_blueprint(oligo_bp, url_prefix='/api/oligo')
    # app.register_blueprint(vienna_bp, url_prefix='/api/vienna')
    # app.register_blueprint(image_bp, url_prefix='/api/image')
    # app.register_blueprint(wordcloud_bp, url_prefix='/api/wordcloud')
    # app.register_blueprint(art_bp, url_prefix='/api/art')
    # app.register_blueprint(chat_bp, url_prefix='/api/chat')

    return app

