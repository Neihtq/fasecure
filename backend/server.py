from flask import Flask, request

from Recognition import Recognition

app = Flask(__name__)
model = Recognition()


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/verify', method=['POST'])
def recognize():
    image = request.form['image']
    name,  access = model.verify(image)
    res = {'name': name, 'access': access}

    return res


@app.route('/register', methods=['PUT'])
def register():
    image = request.form['image']
    name = request.form['name']
    status = model.register(name, image)

    return status


@app.route('/deregister', methods=['DELETE'])
def deregister():
    name = request.form['name']
    status = model.deregister(name)

    return status


@app.route('/wipe', methods=['POST'])
def wipe():
    status = model.wipe_db()

    return status


if __name__ == '__main__':
    app.run(debug=True)


