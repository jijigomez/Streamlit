import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess the data
df = pd.read_csv("newtwitter.csv", encoding='latin1')  # Replace "your_data.csv" with the path to your CSV file
# Perform any necessary preprocessing steps here
df.dropna(subset=['text'], inplace=True)
# Handle missing values in text data
df['text'].fillna('', inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a sentiment analysis model
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = random_forest_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy}")

# Save the model
joblib.dump(random_forest_classifier, "random_forest_classifier.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    # Text input for user to enter new text
    user_input = st.text_input("Enter text to analyze sentiment:")

    # Button to perform sentiment analysis
    if st.button("Analyze Sentiment"):
        if user_input:
            # Vectorize the user input
            user_input_tfidf = tfidf_vectorizer.transform([user_input])

            # Predict sentiment using the trained model
            prediction = random_forest_classifier.predict(user_input_tfidf)[0]

            # Display the sentiment prediction
            st.write(f"Predicted Sentiment: {prediction}")

if __name__ == "__main__":
    main()
