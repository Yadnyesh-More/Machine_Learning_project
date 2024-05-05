from tkinter import *
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
root = Tk()
root.title("Spam Email Predictor ")
root.geometry("700x500")


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

label = Label(root,text="Enter Email : ")
label.place(x=50,y=50)

con = StringVar()

entry = Entry(root,textvariable=con,width=70)
entry.place(x=150,y=55,height=50)

def submit():

    Email_content = con.get()
    transformed_sms = transform_text(Email_content)

    vector_input = tfidf.transform([transformed_sms])

    # USING LOGISTIC REGRESSION CLASSIFIER MODEL 
    result = logistic_model.predict(vector_input)[0]


    # USING ADABOOST CLASSIFIER MODEL 
    result = AdaBoost_model.predict(vector_input)[0]


    # USING KNN CLASSIFER MODEL 
    result = KNN_model.predict(vector_input)[0]


    # USING RANDOM FOREST CLASSIFIER MODEL 
    result = RandomForest_model.predict(vector_input)[0]





    if result == 1:
        print("Spam")

    else:
        print("Not Spam")



button = Button(root,text="submit",command = submit)
button.place(x=350,y=140)



root.mainloop()