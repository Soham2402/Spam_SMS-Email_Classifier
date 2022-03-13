import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer 


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.title("Email/SMS Classifier")
input_text = st.text_input("Enter the message")


ps = PorterStemmer()

def transform_text(text):
    text = text.lower() #Converting it into lower case
    text = nltk.word_tokenize(text) #Tokenizing and seperating all the words
    cleaned_data = []
    for i in text:
        if i.isalnum():  #Only keeping the alpha numeric words in the email/sms
            cleaned_data.append(i)
        
    text = cleaned_data
    cleaned_data = []
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation: #Removing unnceassry words and punctuations
            cleaned_data.append(i)
    
    
    text = cleaned_data
    cleaned_data = []
    
    #Stemming
    for i in text:
        cleaned_data.append(ps.stem(i))
        
    return " ".join(cleaned_data)
    
transformed_text = transform_text(input_text)

vector = tfidf.transform([transformed_text])

result = model.predict(vector)[0]

if st.button("Check"):
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")