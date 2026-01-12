from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

print("Loading dataset and training model...")
ds = load_dataset("codeparrot/codecomplex")
df_train = ds["train"].to_pandas()
df_train.drop(columns=['problem', 'from'], inplace=True)

X = df_train['src']
y = df_train['complexity']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("Model trained successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        code_vec = vectorizer.transform([code])
        prediction = model.predict(code_vec)[0]
        probability = model.predict_proba(code_vec)[0]
        
        classes = model.classes_
        probabilities = {str(cls): float(prob) for cls, prob in zip(classes, probability)}
        
        return jsonify({
            'complexity': str(prediction),
            'probabilities': probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
