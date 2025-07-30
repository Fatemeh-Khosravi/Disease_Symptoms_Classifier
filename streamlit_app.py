# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AI Medical Diagnostic Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    """Load the trained model and preprocessing objects"""
    # Load the MLP model
    mlp_model = tf.keras.models.load_model('mlp_model.h5')
    
    # Load preprocessing objects
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # Load symptom list
    with open('symptoms.pkl', 'rb') as f:
        all_symptoms = pickle.load(f)
    
    return mlp_model, scaler, le, all_symptoms

def get_predictions_with_confidence(mlp_model, symptoms, scaler, le, all_symptoms, top_k=3):
    """Get top k predictions with confidence scores"""
    # Convert symptoms to input vector
    input_vector = [0] * len(all_symptoms)
    for i, symptom in enumerate(all_symptoms):
        if symptom in symptoms:
            input_vector[i] = 1
    
    # Scale the input
    input_scaled = scaler.transform([input_vector])
    
    # Get predictions
    proba = mlp_model.predict(input_scaled, verbose=0)
    
    # Get top k predictions
    top_indices = np.argsort(proba[0])[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        disease = le.inverse_transform([idx])[0]
        conf = proba[0][idx]
        results.append((disease, conf))
    
    return results, proba[0]

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Symptom Disease Classifier</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("ℹ️ About This Project")
    st.sidebar.markdown("""
    **ML-Powered Symptom Analysis**
    
    This application uses a Multi-Layer Perceptron (MLP) neural network trained on medical symptom data to predict potential diseases based on user-reported symptoms.
    
    **Key Features:**
    - 41 different diseases classified
    - 132 symptoms analyzed
    - 91.18% accuracy with partial symptoms
    - Confidence-based predictions
    
    **Disclaimer:**
    This tool is for educational purposes only. Always consult healthcare professionals for medical diagnosis.
    """)
    
    # Load model
    try:
        mlp_model, scaler, le, all_symptoms = load_model_and_data()
    except FileNotFoundError:
        st.error("Model files not found. Please ensure all model files are in the same directory as this app.")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Symptom Selection")
        st.markdown("Select the symptoms you are experiencing (minimum 5 required):")
        
        # Create symptom selection
        selected_symptoms = []
        
        # Group symptoms for better organization
        # Calculate used symptoms first
        general_symptoms = [s for s in all_symptoms if any(word in s.lower() for word in ["fever", "fatigue", "pain", "weakness", "chills"])]
        skin_symptoms = [s for s in all_symptoms if any(word in s.lower() for word in ["skin", "rash", "itching", "blister", "peeling"])]
        respiratory_symptoms = [s for s in all_symptoms if any(word in s.lower() for word in ["cough", "sneezing", "breathing", "chest"])]
        digestive_symptoms = [s for s in all_symptoms if any(word in s.lower() for word in ["stomach", "nausea", "vomiting", "diarrhea", "acidity"])]
        used_symptoms = set()
        used_symptoms.update(general_symptoms)
        used_symptoms.update(skin_symptoms)
        used_symptoms.update(respiratory_symptoms)
        used_symptoms.update(digestive_symptoms)
        symptom_groups = {
            "General Symptoms": [s for s in all_symptoms if any(word in s.lower() for word in ['fever', 'fatigue', 'pain', 'weakness', 'chills'])],
            "Skin Symptoms": [s for s in all_symptoms if any(word in s.lower() for word in ['skin', 'rash', 'itching', 'blister', 'peeling'])],
            "Respiratory Symptoms": [s for s in all_symptoms if any(word in s.lower() for word in ['cough', 'sneezing', 'breathing', 'chest'])],
            "Digestive Symptoms": [s for s in all_symptoms if any(word in s.lower() for word in ['stomach', 'nausea', 'vomiting', 'diarrhea', 'acidity'])],
            "Other Symptoms": [s for s in all_symptoms if s not in used_symptoms]
        }
        
        for group_name, symptoms in symptom_groups.items():
            if symptoms:  # Only show groups that have symptoms
                st.markdown(f"**{group_name}:**")
                cols = st.columns(3)
                for i, symptom in enumerate(symptoms):
                    col_idx = i % 3
                    if cols[col_idx].checkbox(symptom, key=f"{group_name}_{symptom}"):
                        selected_symptoms.append(symptom)
                st.markdown("---")
        
        # Prediction button
        if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
            if len(selected_symptoms) < 5:
                st.error("❌ Please select at least 5 symptoms for prediction.")
            else:
                # Get predictions
                top_predictions, all_proba = get_predictions_with_confidence(
                    mlp_model, selected_symptoms, scaler, le, all_symptoms, top_k=3
                )
                
                # Display results
                st.subheader("🎯 Prediction Results")
                
                # Primary prediction
                primary_disease, primary_confidence = top_predictions[0]
                
                # Confidence color class
                if primary_confidence >= 0.8:
                    conf_class = "confidence-high"
                    conf_emoji = "✅"
                elif primary_confidence >= 0.6:
                    conf_class = "confidence-medium"
                    conf_emoji = "⚠️"
                else:
                    conf_class = "confidence-low"
                    conf_emoji = "❓"
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>{conf_emoji} Most Likely Diagnosis</h3>
                    <h2>{primary_disease}</h2>
                    <p class="{conf_class}">Confidence: {primary_confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Alternative predictions (only if ≥60% confidence)
                alternatives = [(disease, conf) for disease, conf in top_predictions[1:] if conf >= 0.6]
                
                if alternatives:
                    st.markdown("**🤔 Other Possibilities (≥60% confidence):**")
                    for disease, confidence in alternatives:
                        st.markdown(f"• **{disease}** ({confidence:.1%})")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if len(selected_symptoms) < 10:
                    st.info("📋 Consider adding more symptoms for better accuracy")
                if primary_confidence < 0.7:
                    st.warning("🏥 Consider consulting a healthcare professional")
                st.success("✅ This tool is for educational purposes only")
    
    with col2:
        st.subheader("📊 Model Performance")
        
        # Performance metrics
        metrics_data = {
            "Metric": ["Full Test Accuracy", "Partial Input Accuracy", "Diseases Classified", "Symptoms Analyzed"],
            "Value": ["97.62%", "91.18%", "41", "132"]
        }
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        # Confidence distribution chart
        if 'all_proba' in locals():
            st.subheader("📈 Confidence Distribution")
            
            # Create confidence distribution
            conf_data = pd.DataFrame({
                'Disease': [le.inverse_transform([i])[0] for i in range(len(all_proba))],
                'Confidence': all_proba
            }).sort_values('Confidence', ascending=False).head(10)
            
            fig = px.bar(conf_data, x='Confidence', y='Disease', 
                        orientation='h', title="Top 10 Disease Probabilities")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Selected symptoms count
        if selected_symptoms:
            st.subheader("📋 Selected Symptoms")
            st.write(f"**Count:** {len(selected_symptoms)}")
            st.write("**Symptoms:**")
            for symptom in selected_symptoms:
                st.write(f"• {symptom}")

if __name__ == "__main__":
    main() # Force redeploy
