import streamlit as st
import pickle
import string
#import email.parser
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(Message):
    Message = Message.lower()
    Message = nltk.word_tokenize(Message)
    
    y = []
    for i in Message:
        if i.isalnum():
            y.append(i)
    
    Message = y[:]
    y.clear()
    
    for i in Message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    Message = y[:]
    y.clear()
    
    for i in Message:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
logistic_model = pickle.load(open('logistic_model.pkl','rb'))
KNN_model = pickle.load(open('KNN_model.pkl','rb'))
RandomForest_model = pickle.load(open('RandomForest_model.pkl','rb'))
AdaBoost_model = pickle.load(open('AdaBoost_model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = logistic_model.predict(vector_input)[0]
    print(result)
    
    if result == 1:
        st.header("Spam")
        print("spam")

    else:
        st.header("Not Spam ")
        print("Not Spam")
    
    




