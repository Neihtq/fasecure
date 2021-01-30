import numpy as np

from flask import Flask, request, jsonify

from Recognition import Recognition

app = Flask(__name__)
model = Recognition()


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

    return str(status)


@app.route('/deregister', methods=['DELETE'])
def deregister():
    name = request.json['name']
    status = model.deregister(name)

    return status


@app.route('/wipe', methods=['POST'])
def wipe():
    status = model.wipe_db()

    return status


if __name__ == '__main__':
    app.run(debug=True)


