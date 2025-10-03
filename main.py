import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import random
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {"not", "no"}  
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)
tickets = [
  
    "I cannot log into my account",
    "Forgot my password",
    "My login keeps failing",
    "Two-factor authentication is not working",
    "Account locked after too many attempts",
    "My payment failed",
    "I was charged twice",
    "How do I update my card details?",
    "Refund not processed",
    "Billing address is incorrect",
    "App keeps crashing",
    "Website not loading",
    "Page is freezing when I click submit",
    "Error 404 when accessing dashboard",
    "Features are missing after update",
]

labels = [
    "Authentication","Authentication","Authentication","Authentication","Authentication",
    "Payment","Payment","Payment","Payment","Payment",
    "Technical Issue","Technical Issue","Technical Issue","Technical Issue","Technical Issue",
]
MODEL_FILE = "ticket_model.pkl"
if os.path.exists(MODEL_FILE):
    print(" Loading saved model...")
    model = joblib.load(MODEL_FILE)
else:
    print(" Training new model...")
    model = Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=preprocess, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(tickets, labels)
    joblib.dump(model, MODEL_FILE)
    print(" Model trained and saved!")
responses = {
    "Authentication": [
        "Try resetting your password using 'Forgot Password'.",
        "If your account is locked, wait 15 minutes before retrying."
    ],
    "Payment": [
        "Please check if your card details are up to date.",
        "For duplicate charges, contact billing support immediately."
    ],
    "Technical Issue": [
        "Clear your cache and try again.",
        "Restart the app and check for updates."
    ]
}

def answer_ticket(ticket):
    pred = model.predict([ticket])[0]
    response = random.choice(responses.get(pred, ["Sorry, I don’t know the answer."]))
    return f"Category: {pred}\nAnswer: {response}"
while True:
    ticket = input("Enter your ticket ( 'exit' to quit): ")
    if ticket.lower() == 'exit':
        break
    print(answer_ticket(ticket))
    
