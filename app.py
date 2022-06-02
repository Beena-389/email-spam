#app.py
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas

import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

app = Flask(__name__)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

def transform_text(text):
    text=' '.join(text)
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if (i not in stopwords.words('english')) and (i not in string.punctuation):
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)






@app.route('/predict',methods=['POST','GET'])
def predict():
    # receive the values send by user in three text boxes thru request object -> requesst.form.values()
    
    input_sms = [x for x in request.form.values()]
    
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result==0:
        return render_template('index.html', pred='Not-Spam Email')
    elif result==1:
        return render_template('index.html', pred='Spam Email')
    
if __name__ == '__main__':
    app.run(debug=False)











    
        
