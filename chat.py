import pickle, random
vectorizer, clf, intents = pickle.load(open("chatbot_data.pkl", "rb"))

def chatbot_response(msg):
    X = vectorizer.transform([msg])
    tag = clf.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn’t understand that."

print(" Chatbot ready! (type 'quit' to exit)")
while True:
    msg = input("You: ")
    if msg.lower() in ["quit", "exit"]:
        print("Bot: Bye")
        break
    print("Bot:", chatbot_response(msg))
