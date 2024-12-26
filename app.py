import streamlit as st
import pickle
import nltk

def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    nltk.download('stopwords')
    import string
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()

    from nltk.corpus import stopwords

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    stopword = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    for i in text:
        if i not in stopword and i not in punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load models and vectorizers
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    # Preprocessing
    transform_sms = transform(input_sms)

    # Vectorizer
    vector_input = tfidf.transform([transform_sms])

    # Predict
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
