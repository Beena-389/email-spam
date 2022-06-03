#app.py
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas


app = Flask(__name__)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")







@app.route('/predict',methods=['POST','GET'])
def predict():
    # receive the values send by user in three text boxes thru request object -> requesst.form.values()
    
    input_mail = [x for x in request.form.values()]
    
    input_data_features = tfidf.transform(input_mail)
    result = model.predict(input_data_features)
    if result==[0]:
        return render_template('index.html', pred='Spam Email')
    elif result==[1]:
        return render_template('index.html', pred='Not-Spam Email')
    
if __name__ == '__main__':
    app.run(debug=False)











    
        
