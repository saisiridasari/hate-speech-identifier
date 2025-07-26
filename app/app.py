from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "hate_speech_model.pkl"))
vectorizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "tfidf_vectorizer.pkl"))

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    sentence = ""

    if request.method == "POST":
        sentence = request.form.get("sentence", "")
        if sentence.strip():
            vector = vectorizer.transform([sentence])
            result = model.predict(vector)[0]
            prediction = "Hate Speech" if result == 1 else "Not Hate Speech"

    return render_template("index.html", prediction=prediction, sentence=sentence)

if __name__ == "__main__":
    app.run(debug=True)
