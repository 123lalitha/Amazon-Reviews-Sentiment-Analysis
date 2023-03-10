# Library imports
import pandas as pd
import numpy as np
import spacy
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import string
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
import nltk



# Load trained Pipeline
model = joblib.load('sentiment_analysis_model.pkl',"r+")


stopwords = list(STOP_WORDS)

# Create the app object
app = Flask(__name__)




# creating a function for data cleaning
from custom_tokenizer_function import CustomTokenizer
reviewer_name=[]
reviewer_city=[]
review=[]
sentiment=[]
category=[]
product=[]
# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST',"GET"])
def predict():
    new_review = [str(x) for x in request.form.values()]

    reviewer_name.append(new_review[0])
    reviewer_city.append(new_review[1])
    category.append(new_review[2])
    product.append(new_review[3])
    review.append(new_review[4])
    
#     data = pd.DataFrame(new_review)
#     data.columns = ['new_review']
    predictions = model.predict([str(new_review)])[0]
    sentiment.append(predictions)
    data=pd.DataFrame(zip(reviewer_name,reviewer_city,category,product,review,sentiment))
    data.to_csv('app_reviews',mode="a+",header=False)


    if predictions == "Positive":
              
        return render_template('index.html', prediction_text='Positive 🙂')
    elif predictions=="Neutral":
        
        return render_template('index.html', prediction_text='Neutral 😐')
    else:
        
        return render_template('index.html', prediction_text='Negative 😔')

    
name=[] 
email=[]
subject=[]
message=[] 
@app.route('/contact',methods=['POST',"GET"])
def contact():
    details = [str(x) for x in request.form.values()]

    name.append(details[0])
    email.append(details[1])
    subject.append(details[2])
    message.append(details[3])
    contact_details=pd.DataFrame(zip(name,email,subject,message))
    contact_details.to_csv('contact_details',mode="a+",header=False)

    return render_template('index.html', contact_dts='Thank You, We Will Get Back To You Soon!')

   


if __name__ == "__main__":
    app.run(debug=True)