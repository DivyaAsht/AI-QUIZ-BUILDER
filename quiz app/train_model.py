import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


# Load from SQLite
conn = sqlite3.connect('quiz_data.db')
df = pd.read_sql("SELECT * FROM training_questions", conn)
conn.close()

# Vectorize the question text
vectorizer = TfidfVectorizer(max_features=300)
X = vectorizer.fit_transform(df['question']).toarray()
y = df['difficulty']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Print performance
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'model/difficulty_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("✅ Model saved!")
