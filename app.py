# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:59:19 2022

@author: USER
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.externals
import pickle

# load the model from disk
filename = 'spamclassifier.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('spamdata.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
  
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('predict.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)