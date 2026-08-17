import os

from app import create_app

app = create_app()

if __name__ == '__main__':
    # Debug is opt-in: Werkzeug's debugger allows arbitrary code execution on
    # any host that can reach it, so it must never be the default.
    #   FLASK_DEBUG=1 python run.py   → reloader + debugger
    debug = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes')
    host  = os.environ.get('HOST', '127.0.0.1')
    port  = int(os.environ.get('PORT', 5001))
    app.run(debug=debug, host=host, port=port)
