import base64
import io
import optparse
import os

import cv2
import flask
import numpy as np
import requests
from PIL import Image

import forward

UPLOAD_FOLDER = '/tmp/fbp_demo'
ALLOWED_EXTENSIONS = {'bmp', 'jpeg', 'jpg', 'jpe', 'png', 'pbm', 'tif'}

app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        byte_stream = io.BytesIO(requests.get(imageurl).content)
        image = cv2.imdecode(np.frombuffer(byte_stream.read(), np.uint8), 1)

    except:
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    result = app.predictor.predict(image)
    return flask.render_template(
        'index.html', has_result=True,
        result=[(data_uri_encoder(face_img), f"{output:.2f}") for face_img, output in result]
    )


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        imagefile = flask.request.files['imagefile']
        byte_stream = io.BytesIO()
        imagefile.save(byte_stream)
        pimg = Image.open(byte_stream)
        image = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    except Exception as err:
        print(err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.predictor.predict(image)
    return flask.render_template(
        'index.html', has_result=True,
        result=[(data_uri_encoder(face_img), f"{output:.2f}") for face_img, output in result]
    )


def data_uri_encoder(face_img: np.array):
    image_pil = Image.fromarray(face_img)
    byte_stream = io.BytesIO()
    image_pil.save(byte_stream, format='png')
    data = base64.b64encode(byte_stream.getvalue()).decode("utf-8")
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return '.' in filename and filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS


def start_from_terminal(app):
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)
    parser.add_option(
        '-r', '--resnet',
        help="use resnet",
        action='store_true', default=False)

    opts, args = parser.parse_args()

    app.predictor = forward.Predictor(opts.gpu)
    if opts.resnet:
        app.predictor.use_resnet()
    else:
        app.predictor.use_alexnet()

    app.run(debug=opts.debug, host='127.0.0.1', port=opts.port)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
