import numpy as np

from flask import Flask, request, jsonify
from face_recognition.evaluate import evaluate

from Recognition import Recognition

app = Flask(__name__)
#model = Recognition()


@app.route('/')
def hello_world():
    print("TESTESTSEET")
    return 'Hello World!'


@app.route('/verify', methods=['POST'])
def recognize():
    data = request.json
    image = np.array(data['image'])
    name,  access = model.verify(image)
    res = {'name': name, 'access': access}

    return jsonify(res)


@app.route('/register', methods=['PUT', 'POST'])
def register():
    data = request.json
    image = np.array(data['image'])
    name = data['name']
    status = model.register(name, image)
    res = {'status': status}

    return jsonify(res)


@app.route('/deregister', methods=['DELETE'])
def deregister():
    name = request.json['name']
    status = model.deregister(name)
    res = {'status': status}

    return jsonify(res)


@app.route('/wipe', methods=['POST'])
def wipe():
    status = model.wipe_db()
    res = {'status': status}

    return jsonify(res)


@app.route('/listAll', methods=['GET'])
def list_all():
    label_list = model.list_labels()
    res = {'names': label_list}

    return jsonify(res)


if __name__ == '__main__':
     #app.run(host='0.0.0.0', debug=True)
     evaluate()