import joblib
import flask
import DiseaseDetection
import cv2
from skimage import io
import os
from flask import request
from flask import jsonify

app = flask.Flask(__name__)
UPLOAD_FOLDER_DISEASE = "./mysite/disease"
app.config["UPLOAD_FOLDER_DISEASE"] = UPLOAD_FOLDER_DISEASE


@app.route('/predict/apple-disease-detection', methods=['POST'])
def predict():
    dict = {}
    if "fruit" not in request.files:
            return "there is no file1 in form!"
    type = request.form.get('type')
    imagefile = flask.request.files.get('fruit', '')
    path = os.path.join(app.config["UPLOAD_FOLDER_DISEASE"], imagefile.filename)
    imagefile.save(path)
    img = io.imread(path, 0)[:, :, 0:3]

    classifier = None
    if type == 'apple':
        classifier = joblib.load('./mysite/models/Apple.joblib')
    elif type == 'cherry':
        classifier = joblib.load('./mysite/models/cherry.joblib')
    elif type == 'corn':
        classifier = joblib.load('./mysite/models/corn.joblib')
    elif type == 'grape':
        classifier = joblib.load('./mysite/models/grape.joblib')
    elif type == 'peach':
        classifier = joblib.load('./mysite/models/peach.joblib')
    elif type == 'pepper':
        classifier = joblib.load('./mysite/models/pepper.joblib')
    elif type == 'potato':
        classifier = joblib.load('./mysite/models/potato.joblib')
    elif type == 'strawberry':
        classifier = joblib.load('./mysite/models/strawberry.joblib')
    elif type == 'tomato':
        classifier = joblib.load('./mysite/models/Tomato.joblib')

    prediction = classifier.predict(DiseaseDetection.extract_all([img]))
    dict['prediction'] = DiseaseDetection.dictionary[int(prediction[0])]
    return jsonify(dict)


@app.route('/', methods=['GET'])
def get():
    return jsonify("Hello World!")

if __name__ == 'main':
    app.run(port=8080)