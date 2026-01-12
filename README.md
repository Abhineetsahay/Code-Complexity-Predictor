# Code Complexity Predictor

A machine learning web application that predicts the time complexity of code snippets using Natural Language Processing and Logistic Regression.

## Features

- **Real-time Prediction**: Analyze code complexity instantly
- **Multiple Complexity Classes**: Predicts Constant, Linear, Quadratic, and Cubic complexities
- **Confidence Scores**: Shows probability distribution across all complexity classes
- **Simple UI**: Clean, intuitive interface for easy code analysis
- **RESTful API**: JSON API endpoint for integration with other tools

## Technologies Used

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, TF-IDF Vectorization, Logistic Regression
- **Dataset**: CodeParrot CodeComplex dataset
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Gunicorn, Docker


## Usage

### Web Interface

1. Paste your code into the text area
2. Click "Analyze Complexity"
3. View the predicted complexity and confidence scores

### API Endpoint

**POST** `/predict`

**Request Body:**
```json
{
  "code": "your code here"
}
```

**Response:**
```json
{
  "complexity": "linear",
  "probabilities": {
    "constant": 0.15,
    "linear": 0.65,
    "quadratic": 0.15,
    "cubic": 0.05
  }
}
```

## Model Details

- **Algorithm**: Logistic Regression
- **Feature Extraction**: TF-IDF Vectorization (max 5000 features)
- **Dataset**: CodeParrot CodeComplex dataset
- **Train/Test Split**: 80/20
- **Complexity Classes**: Constant, Linear, Quadratic, Cubic

