import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

X, y = [], []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        X.append(pattern)
        y.append(intent["tag"])

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)
clf = LogisticRegression(max_iter=200)
clf.fit(X_tfidf, y)
with open("chatbot_data.pkl", "wb") as f:
    pickle.dump((vectorizer, clf, intents), f)

print(" Logistic Regression chatbot trained and saved (chatbot_data.pkl)")
