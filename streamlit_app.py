import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import os

# Page configuration
st.set_page_config(
    page_title="Symptom Disease Classifier",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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
    try:
        # Load the MLP model
        mlp_model = tf.keras.models.load_model('mlp_model.h5')
        
        # Load preprocessing objects
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('symptoms.pkl', 'rb') as f:
            all_symptoms = pickle.load(f)
        
        return mlp_model, scaler, label_encoder, all_symptoms
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def create_symptom_groups(all_symptoms):
    """Create symptom groups for better organization"""
    # Define symptom categories
    groups = {
        "General Symptoms": ["fever", "fatigue", "headache", "nausea", "vomiting", "dizziness", "weakness"],
        "Respiratory": ["cough", "sore throat", "runny nose", "chest pain", "shortness of breath", "wheezing"],
        "Digestive": ["abdominal pain", "diarrhea", "constipation", "bloating", "loss of appetite", "heartburn"],
        "Skin": ["rash", "itching", "swelling", "redness", "blisters", "dry skin"],
        "Musculoskeletal": ["joint pain", "back pain", "muscle pain", "stiffness", "swelling"],
        "Neurological": ["numbness", "tingling", "seizures", "memory loss", "confusion"],
        "Cardiovascular": ["chest pain", "irregular heartbeat", "high blood pressure", "palpitations"],
        "Urinary": ["frequent urination", "painful urination", "blood in urine", "incontinence"]
    }
    
    # Filter groups to only include symptoms that exist in our dataset
    filtered_groups = {}
    for group_name, symptoms in groups.items():
        filtered_symptoms = [s for s in symptoms if s in all_symptoms]
        if filtered_symptoms:
            filtered_groups[group_name] = filtered_symptoms
    
    # Add remaining symptoms to "Other Symptoms"
    used_symptoms = set()
    for symptoms in filtered_groups.values():
        used_symptoms.update(symptoms)
    
    other_symptoms = [s for s in all_symptoms if s not in used_symptoms]
    if other_symptoms:
        filtered_groups["Other Symptoms"] = other_symptoms
    
    return filtered_groups

def get_predictions_with_confidence(model, scaler, label_encoder, selected_symptoms, all_symptoms):
    """Get predictions with confidence scores"""
    try:
        # Create input vector
        input_vector = np.zeros(len(all_symptoms))
        for symptom in selected_symptoms:
            if symptom in all_symptoms:
                idx = all_symptoms.index(symptom)
                input_vector[idx] = 1
        
        # Reshape for model
        input_vector = input_vector.reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_vector)
        
        # Get predictions
        predictions = model.predict(input_scaled, verbose=0)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            disease = label_encoder.inverse_transform([idx])[0]
            confidence = predictions[0][idx] * 100
            results.append({
                'disease': disease,
                'confidence': confidence,
                'rank': i + 1
            })
        
        return results
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Main app
def main():
    st.markdown('<h1 class="main-header">🩺 Symptom Disease Classifier</h1>', unsafe_allow_html=True)
    
    # Load model and data
    mlp_model, scaler, label_encoder, all_symptoms = load_model_and_data()
    
    if mlp_model is None:
        st.error("❌ Failed to load the model. Please check if all model files are present.")
        st.info("Required files: mlp_model.h5, scaler.pkl, label_encoder.pkl, symptoms.pkl")
        st.warning("⚠️ The app will work in demo mode without the ML model.")
        
        # Demo mode with sample symptoms
        demo_symptoms = [
            "fever", "headache", "fatigue", "cough", "sore throat",
            "nausea", "vomiting", "abdominal pain", "diarrhea", "rash",
            "joint pain", "back pain", "chest pain", "shortness of breath"
        ]
        
        st.sidebar.header("🔍 Select Your Symptoms (Demo Mode)")
        st.sidebar.write("Select at least 5 symptoms for demo prediction:")
        
        selected_symptoms = []
        for symptom in demo_symptoms:
            if st.sidebar.checkbox(symptom.replace('_', ' ').title(), key=f"demo_{symptom}"):
                selected_symptoms.append(symptom)
        
        # Demo prediction
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header("📋 Selected Symptoms")
            if selected_symptoms:
                symptom_text = ", ".join([s.replace('_', ' ').title() for s in selected_symptoms])
                st.info(f"**{len(selected_symptoms)} symptoms selected:** {symptom_text}")
                
                st.header("🔍 Demo Prediction Results")
                if len(selected_symptoms) < 5:
                    st.warning("⚠️ Please select at least 5 symptoms for demo prediction.")
                else:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### 🎯 **Demo Diagnosis**")
                    st.markdown(f"**Disease:** Common Cold (Demo)")
                    st.markdown(f"**Confidence:** <span class='confidence-medium'>75.5%</span>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("### 🔍 **Demo Alternative Diagnoses**")
                    st.markdown("**Flu** - 65.2% confidence")
                    st.markdown("**Sinusitis** - 60.1% confidence")
            else:
                st.info("👈 Please select symptoms from the sidebar to get started.")
        
        with col2:
            st.header("ℹ️ Demo Information")
            st.write("""
            **Demo Mode Active:**
            - This is a demonstration without the ML model
            - Add model files to enable real predictions
            - Select at least 5 symptoms for demo
            
            **Note:** This is for educational purposes only.
            """)
            
            if selected_symptoms:
                st.metric("Symptoms Selected", len(selected_symptoms))
                if len(selected_symptoms) >= 5:
                    st.success("✅ Ready for demo prediction")
                else:
                    st.warning(f"⚠️ Need {5 - len(selected_symptoms)} more symptoms")
        
        return
    
    
    # Create symptom groups
    symptom_groups = create_symptom_groups(all_symptoms)
    
    # Sidebar for symptom selection
    st.sidebar.header("🔍 Select Your Symptoms")
    st.sidebar.write("Select at least 5 symptoms for accurate prediction:")
    
    selected_symptoms = []
    
    # Create expandable sections for each symptom group
    for group_name, symptoms in symptom_groups.items():
        with st.sidebar.expander(f"📁 {group_name} ({len(symptoms)} symptoms)"):
            for symptom in symptoms:
                if st.checkbox(symptom.replace('_', ' ').title(), key=f"checkbox_{symptom}"):
                    selected_symptoms.append(symptom)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Selected Symptoms")
        if selected_symptoms:
            # Display selected symptoms
            symptom_text = ", ".join([s.replace('_', ' ').title() for s in selected_symptoms])
            st.info(f"**{len(selected_symptoms)} symptoms selected:** {symptom_text}")
            
            # Prediction section
            st.header("🔍 Prediction Results")
            
            if len(selected_symptoms) < 5:
                st.warning("⚠️ Please select at least 5 symptoms for accurate prediction.")
            else:
                # Get predictions
                predictions = get_predictions_with_confidence(
                    mlp_model, scaler, label_encoder, selected_symptoms, all_symptoms
                )
                
                if predictions:
                    # Display primary prediction
                    primary = predictions[0]
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### 🎯 **Primary Diagnosis**")
                    st.markdown(f"**Disease:** {primary['disease']}")
                    
                    # Color-code confidence
                    if primary['confidence'] >= 80:
                        confidence_class = "confidence-high"
                    elif primary['confidence'] >= 60:
                        confidence_class = "confidence-medium"
                    else:
                        confidence_class = "confidence-low"
                    
                    st.markdown(f"**Confidence:** <span class='{confidence_class}'>{primary['confidence']:.1f}%</span>", 
                              unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show alternative predictions if confidence is high enough
                    if len(predictions) > 1:
                        alternatives = [p for p in predictions[1:] if p['confidence'] >= 60]
                        if alternatives:
                            st.markdown("### 🔍 **Alternative Diagnoses**")
                            st.markdown("*Showing only if confidence ≥ 60%*")
                            
                            for alt in alternatives:
                                st.markdown(f"**{alt['disease']}** - {alt['confidence']:.1f}% confidence")
        else:
            st.info("👈 Please select symptoms from the sidebar to get started.")
    
    with col2:
        st.header("ℹ️ Information")
        st.write("""
        **How to use:**
        1. Select at least 5 symptoms from the sidebar
        2. The model will predict the most likely disease
        3. Alternative diagnoses are shown if confidence ≥ 60%
        
        **Note:** This is for educational purposes only. Always consult a healthcare professional for medical advice.
        """)
        
        if selected_symptoms:
            st.metric("Symptoms Selected", len(selected_symptoms))
            if len(selected_symptoms) >= 5:
                st.success("✅ Ready for prediction")
            else:
                st.warning(f"⚠️ Need {5 - len(selected_symptoms)} more symptoms")

if __name__ == "__main__":
    main()
