# 🩺 Symptom Disease Classifier

A machine learning-powered web application that predicts potential diseases based on user-reported symptoms using a Multi-Layer Perceptron (MLP) neural network.

## 🚀 Features

- **MLP Neural Network**: Advanced deep learning model for accurate disease prediction
- **Interactive UI**: User-friendly Streamlit interface with symptom categorization
- **Partial Input Support**: Works with as few as 5 symptoms (91.18% accuracy)
- **Confidence Scoring**: Shows prediction confidence and alternative diagnoses
- **41 Diseases**: Classifies across a wide range of medical conditions
- **132 Symptoms**: Comprehensive symptom database

## �� Model Performance

- **Full Test Accuracy**: 97.62%
- **Partial Input Accuracy** (5 symptoms): 91.18%
- **Diseases Classified**: 41
- **Symptoms Analyzed**: 132

## 🛠️ Technology Stack

- **Backend**: Python, TensorFlow, Scikit-learn
- **Frontend**: Streamlit
- **Deployment**: Railway
- **Model**: Multi-Layer Perceptron (MLP) Neural Network

## 📁 Project Structure

```
Symptoms-classifier/
├── streamlit_app.py      # Main Streamlit application
├── requirements.txt      # Python dependencies
├── runtime.txt          # Python version specification
├── Procfile            # Railway deployment configuration
├── README.md           # Project documentation
├── mlp_model.h5        # Trained MLP model
├── scaler.pkl          # Feature scaler
├── label_encoder.pkl   # Label encoder
└── symptoms.pkl        # Symptom list
```

## 🚀 Deployment

### Railway Deployment

1. **Fork/Clone** this repository
2. **Connect** to Railway
3. **Deploy** automatically
4. **Access** your live app

### Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd Symptoms-classifier

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

## 🎯 How to Use

1. **Select Symptoms**: Choose at least 5 symptoms from the categorized sidebar
2. **Get Prediction**: The model will predict the most likely disease
3. **Review Confidence**: Check the confidence score and alternative diagnoses
4. **Consult Professional**: Always seek medical advice for actual diagnosis

## ⚠️ Important Disclaimers

- **Educational Purpose Only**: This tool is designed for educational and demonstration purposes
- **Not Medical Advice**: Predictions should not replace professional medical consultation
- **Accuracy Limitations**: While the model performs well, it has limitations and should not be used for actual medical diagnosis
- **Data Privacy**: No personal health data is stored or transmitted

## 🔒 Safety Features

- **Minimum Symptom Requirement**: Requires at least 5 symptoms for prediction
- **Confidence Thresholds**: Only shows alternative diagnoses with ≥60% confidence
- **Professional Consultation**: Always recommends consulting healthcare professionals
- **Educational Focus**: Clearly marked as educational tool

## 📈 Model Training

The MLP model was trained on a comprehensive dataset with:
- **Training Data**: 4,920 samples
- **Test Data**: 1,230 samples
- **Features**: 132 binary symptom indicators
- **Classes**: 41 different diseases
- **Architecture**: 3 hidden layers with dropout regularization

## 🤝 Contributing

This is a portfolio project demonstrating machine learning and web development skills. For educational purposes only.

## 📄 License

This project is for educational and portfolio purposes only.

---

**⚠️ Medical Disclaimer**: This application is for educational purposes only and should not be used for actual medical diagnosis. Always consult with qualified healthcare professionals for medical advice and diagnosis.
