from flask import Flask, render_template, request
import numpy as np
import pickle

apph = Flask(__name__ ,template_folder='template')
modelh = pickle.load(open('modelh.pkl', 'rb'))


@apph.route('/')
def index():
    return render_template('index.html')


@apph.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['bedrooms']
    val2 = request.form['bathrooms']
    val3 = request.form['floors']
    val4 = request.form['yr_built']
    arr = np.array([val1, val2, val3, val4])
    arr = arr.astype(np.float64)
    pred = modelh.predict([arr])

    return render_template('index.html', data=int(pred))


if __name__ == '__main__':
    apph.run(debug=True)