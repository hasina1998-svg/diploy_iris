from flask import Flask, render_template, request
import pickle
import numpy as np
from iris import accuracy,set_matrix,ver_matrix,vir_matrix

model = pickle.load(open('iris.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def pred():
    data1 = request.form['sl']
    data2 = request.form['sw']
    data3 = request.form['pl']
    data4 = request.form['pw']
    arr = np.array([[data1, data2, data3, data4]])
    output = model.predict(arr)
    return render_template('output.html', data=output)


@app.route('/info')
def info():
    return render_template('info.html',a= accuracy,s=set_matrix, v=ver_matrix,i=vir_matrix)

if __name__ == "__main__":
    app.run(debug=True)
