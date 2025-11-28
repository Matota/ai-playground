import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Create a simple dataset (simulating a CSV load)
data = {
    'text': [
        "I loved this movie, it was fantastic!",
        "The plot was boring and the acting was bad.",
        "What a great film, highly recommended.",
        "I fell asleep halfway through, terrible.",
        "The cinematography was beautiful, but the story was weak.",
        "An absolute masterpiece of cinema.",
        "Waste of time, do not watch.",
        "I enjoyed it, good fun.",
        "Not my cup of tea, but well made.",
        "Worst movie I have ever seen."
    ],
    'sentiment': [
        'positive', 'negative', 'positive', 'negative', 'negative', 
        'positive', 'negative', 'positive', 'positive', 'negative'
    ]
}

print("Loading data...")
df = pd.DataFrame(data)
print(f"Data loaded. Shape: {df.shape}")
print(df.head())

# 2. Preprocessing & Vectorization
print("\nVectorizing text...")
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Train Model
print("Training model...")
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Evaluate
print("Evaluating...")
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Test with new custom sentences
print("\nTesting with new sentences:")
new_sentences = [
    "This was the best movie ever!",
    "I hated every minute of it."
]
new_X = vectorizer.transform(new_sentences)
predictions = model.predict(new_X)

for text, pred in zip(new_sentences, predictions):
    print(f"Text: '{text}' -> Sentiment: {pred}")
