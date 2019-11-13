from flask import Flask, request, render_template, jsonify

app = Flask(__name__, static_folder='', template_folder='', static_url_path='')

@app.route('/')
def homepage():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run()
