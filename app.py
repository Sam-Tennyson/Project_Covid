from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from keras_preprocessing import image

from werkzeug.utils import secure_filename
import os
import numpy as np

app = Flask(__name__)
model = load_model("VGG16_Model.hdf5")


def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        # loc = request.form['loc']
        d = {1: fname, 2: lname, 3: email}
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/image/', secure_filename(f.filename))
        # print(basepath)
        # print(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        # x = x.reshape([64, 64]);
        disease_class = ['Covid-19 +ve', 'Covid-19 -ve']
        a = preds[0]
        ind = np.argmax(a)
        print('Prediction:', disease_class[ind])
        result = disease_class[ind]
        return render_template("result.html", d=d, result=result, covid_positive=preds[0][0], covid_negative=preds[0][1])
    return None


if __name__ == '__main__':
    app.run(debug=True, port=7000)
