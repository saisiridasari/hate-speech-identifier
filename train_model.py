# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
df = pd.read_csv("dataset/labeled_data.csv")

# 2. Prepare data
X = df["tweet"]          # or 'text' if column name is different
y = df["class"]          # assuming 'class' is the label (0 = normal, 1 = hate)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)

# 5. Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 6. Save model and vectorizer
joblib.dump(model, "models/hate_speech_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved to /models/")
