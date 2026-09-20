import os
from flask import Flask, redirect, url_for

from config import Config
from models import db


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Make sure the SQLite file's parent directory exists, whatever the URI
    # actually resolves to (absolute path by default; a relative override in
    # .env is also handled, since os.makedirs below uses whatever path is in
    # the URI rather than assuming a fixed "instance" folder).
    uri = app.config["SQLALCHEMY_DATABASE_URI"]
    if uri.startswith("sqlite:///"):
        db_path = uri.replace("sqlite:///", "", 1)
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    db.init_app(app)

    from routes.chat import chat_bp
    from routes.dashboard import dashboard_bp
    app.register_blueprint(chat_bp)
    app.register_blueprint(dashboard_bp)

    @app.route("/")
    def index():
        return redirect(url_for("dashboard.home"))

    with app.app_context():
        db.create_all()

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=Config.DEBUG, port=5000)
