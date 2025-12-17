from flask import Flask, request, jsonify, render_template, url_for, redirect
import numpy as np
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)



@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/result',methods=['GET','POST'])
def predict():
   # age,sex,cp,,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
    # Retrieve data from URL parameters
    age = request.form.get('age')
    s = request.form.get('sex')
    sex = 1 if s.lower() == 'male' else 0
    cp = request.form.get('cp')
    trestbps = request.form.get('trestbps')
   # chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')
    form_data_array = [
    age,
    sex,
    cp,
    trestbps,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]
    my_array=np.array(form_data_array).reshape(-1, 12)
    new_data = my_array.astype(float)  
    
    prediction = loaded_model.predict(new_data)
    out=np.round(prediction[0],2)
    
    if out == 1: 
        risklevel='low'
    else:
        risklevel='high'
    
    return render_template('result.html',form_data=risklevel)

if __name__ == '__main__':
    app.run(debug=True)
