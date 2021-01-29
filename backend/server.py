from flask import Flask, request
from face_recognition.evaluate import evaluate


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/recognize')
def recognize():
    return 'Recognition!'


@app.route('/register')
def register():
    return 'Registration!'


@app.route('/deregister')
def deregister():
    return 'Deregistration!'


@app.route('/wipe', methods=['POST'])
def wipe():
    password = request.form['password']
    username = request.form['username']
    return f'Wipe Database! {username} and {password}'


if __name__=='__main__':
    evaluate()
    #app.run(debug=True)