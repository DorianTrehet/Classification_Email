from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load pre-trained models
cv = pickle.load(open("models/cv.pkl", 'rb'))
clf = pickle.load(open("models/clf.pkl", 'rb'))

@app.route('/')
def home():
    return render_template("input.html")

@app.route('/predict', methods=['POST'])
def predict():
    email_content = request.form['email']
    tokenized_email = cv.transform([email_content])
    prediction = clf.predict(tokenized_email)
    result = "Spam" if prediction[0] == 1 else "Non Spam"
    return render_template('input.html', resultat=f"{result}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    email_content = data['email']
    tokenized_email = cv.transform([email_content])
    prediction = clf.predict(tokenized_email)
    result = "Spam" if prediction[0] == 1 else "Non Spam"
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
