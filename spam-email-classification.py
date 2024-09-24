from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample email data (content, label)
emails = [
    ("Get rich quick! Buy now!", "spam"),
    ("Meeting scheduled for tomorrow", "ham"),
    ("Congratulations! You've won a prize", "spam"),
    ("Project update: new features implemented", "ham")
]

# Separate content and labels
X, y = zip(*emails)

# Create a pipeline with text vectorization and Naive Bayes classifier
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X, y)

# Classify new emails
new_emails = ["Free money! Click here!", "Weekly team meeting agenda", "Get rich as a developer quick "]
predictions = pipeline.predict(new_emails)

for email, prediction in zip(new_emails, predictions):
    print(f"Email: {email}")
    print(f"Prediction: {prediction}\n")
