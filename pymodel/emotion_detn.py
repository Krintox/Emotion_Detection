from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)

nltk.download('stopwords')
nltk.download('wordnet')

train_data = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])
val_data = pd.read_csv('val.txt', sep=';', header=None, names=['text', 'emotion'])
test_data = pd.read_csv('test.txt', sep=';', header=None, names=['text', 'emotion'])

train_val_data = pd.concat([train_data, val_data])

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    processed_text = ' '.join(words)
    return processed_text


train_val_data['processed_text'] = train_val_data['text'].apply(preprocess_text)
test_data['processed_text'] = test_data['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X_train_val = vectorizer.fit_transform(train_val_data['processed_text'])
X_test = vectorizer.transform(test_data['processed_text'])
y_train_val = train_val_data['emotion']

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

val_predictions = classifier.predict(X_val)

val_accuracy = accuracy_score(y_val, val_predictions)
print('Validation accuracy:', val_accuracy)

test_predictions = classifier.predict(X_test)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = classifier.predict(vectorized_text)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
