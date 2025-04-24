import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Set page config with attractive styling
st.set_page_config(
    page_title="DermaScan AI",
    page_icon=":skin-tone-3:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .header-text {
        color: #2e86de;
        text-align: center;
        font-size: 2.5rem !important;
    }
    .disease-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .disease-title {
        color: #2e86de;
        border-bottom: 2px solid #2e86de;
        padding-bottom: 5px;
    }
    .prediction-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .confidence-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background-color: #2e86de;
        text-align: center;
        color: white;
        font-size: 12px;
        line-height: 20px;
    }
    .upload-box {
        border: 2px dashed #2e86de;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the model (cache to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('skin_disease_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Disease information
disease_info = {
    'Acne and Rosacea': {
        'description': "Acne is a skin condition that occurs when hair follicles become clogged with oil and dead skin cells. Rosacea is a chronic skin condition that causes redness and visible blood vessels.",
        'symptoms': ["Pimples", "Blackheads/whiteheads", "Redness", "Visible blood vessels"],
        'treatment': ["Topical treatments", "Oral medications", "Laser therapy"],
        'icon': "🧴"
    },
    'Actinic Keratosis and Malignant Lesions': {
        'description': "Actinic keratosis is a rough, scaly patch on the skin that develops from years of sun exposure. It can develop into skin cancer if left untreated.",
        'symptoms': ["Rough, scaly patches", "Raised bumps", "Redness", "Skin crusting"],
        'treatment': ["Cryotherapy", "Topical medications", "Photodynamic therapy"],
        'icon': "☀️"
    },
    'Atopic Dermatitis': {
        'description': "Also known as eczema, this condition makes skin red and itchy. It's common in children but can occur at any age.",
        'symptoms': ["Dry skin", "Itching", "Red to brownish-gray patches", "Small raised bumps"],
        'treatment': ["Moisturizers", "Prescription creams", "Oral drugs", "Light therapy"],
        'icon': "🦠"
    },
    'Exanthems and Drug Eruptions': {
        'description': "Skin reactions that occur as a result of medication or viral infections.",
        'symptoms': ["Rash", "Fever", "Itching", "Skin redness"],
        'treatment': ["Discontinuing causative drug", "Antihistamines", "Corticosteroids"],
        'icon': "💊"
    },
    'Hair Loss Diseases': {
        'description': "Conditions that result in hair loss, including alopecia areata and androgenetic alopecia.",
        'symptoms': ["Patchy hair loss", "Thinning hair", "Complete hair loss", "Scaly scalp"],
        'treatment': ["Minoxidil", "Finasteride", "Hair transplant", "Steroid injections"],
        'icon': "💇"
    },
    'Herpes HPV and STDs': {
        'description': "Viral infections that can cause skin manifestations including herpes simplex and human papillomavirus (HPV).",
        'symptoms': ["Blisters", "Warts", "Ulcers", "Itching"],
        'treatment': ["Antiviral medications", "Topical treatments", "Vaccines"],
        'icon': "⚠️"
    },
    'Nail Fungus': {
        'description': "A fungal infection that affects the nails, usually starting as a white or yellow spot under the tip of the nail.",
        'symptoms': ["Thickened nails", "Discoloration", "Brittle texture", "Distorted shape"],
        'treatment': ["Antifungal medications", "Topical treatments", "Laser therapy"],
        'icon': "💅"
    },
    'Poison Ivy and Contact Dermatitis': {
        'description': "An allergic reaction caused by contact with certain substances like poison ivy, nickel, or cosmetics.",
        'symptoms': ["Red rash", "Itching", "Blisters", "Swelling"],
        'treatment': ["Avoiding allergen", "Topical steroids", "Antihistamines", "Cool compresses"],
        'icon': "🌿"
    },
    'Psoriasis and Lichen Planus': {
        'description': "Psoriasis is an autoimmune condition that causes rapid skin cell buildup. Lichen planus is an inflammatory condition.",
        'symptoms': ["Scaly patches", "Itching", "Purple bumps", "White lacy patches"],
        'treatment': ["Topical treatments", "Light therapy", "Systemic medications"],
        'icon': "🔴"
    }
}

# Prediction function
def predict_image(img, model, img_height=180, img_width=180):
    """Predict skin disease from uploaded image"""
    class_names = list(disease_info.keys())
    
    try:
        # Preprocess image
        img = img.resize((img_height, img_width))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)
        
        return predicted_class, confidence, predictions
    
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None, None

# Main app
def main():
    st.markdown('<h1 class="header-text">DermaScan AI</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px; color: #555;">
        A deep learning-powered skin disease classification system that identifies 9 common conditions
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Disease Encyclopedia", "🔍 Image Classifier", "ℹ️ About"])
    
    with tab1:
        st.header("Skin Disease Information")
        st.write("Learn about the 9 skin conditions this model can identify:")
        
        # Display disease information in cards
        for disease, info in disease_info.items():
            with st.container():
                st.markdown(f'<div class="disease-card">', unsafe_allow_html=True)
                
                # Disease header with icon
                st.markdown(f'<h3 class="disease-title">{info["icon"]} {disease}</h3>', unsafe_allow_html=True)
                
                # Description
                st.write(info['description'])
                
                # Symptoms and treatment in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Common Symptoms:**")
                    for symptom in info['symptoms']:
                        st.write(f"- {symptom}")
                
                with col2:
                    st.markdown("**Treatment Options:**")
                    for treatment in info['treatment']:
                        st.write(f"- {treatment}")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("Skin Disease Classifier")
        st.write("Upload an image of a skin condition to get a prediction")
        
        # Upload box with custom styling
        with st.container():
            st.markdown('<div class="upload-box">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display the uploaded image
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=300)
                
                # Make prediction when button is clicked
                if st.button("Analyze Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        predicted_class, confidence, predictions = predict_image(image, model)
                    
                    if predicted_class:
                        # Prediction results card
                        with st.container():
                            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                            
                            # Prediction result
                            st.success(f"**Prediction:** {predicted_class}  \n**Confidence:** {confidence:.2%}")
                            
                            # Confidence bars for all classes
                            st.markdown("**Prediction Confidence Levels:**")
                            class_names = list(disease_info.keys())
                            sorted_indices = np.argsort(predictions)[::-1]  # Sort descending
                            
                            for i in sorted_indices:
                                confidence_percent = predictions[i] * 100
                                st.write(f"{class_names[i]}")
                                st.markdown(
                                    f'<div class="confidence-bar">'
                                    f'<div class="confidence-fill" style="width: {confidence_percent}%">{confidence_percent:.1f}%</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            
                            # Show information about the predicted disease
                            st.markdown("---")
                            st.subheader(f"About {predicted_class}")
                            info = disease_info[predicted_class]
                            st.write(info['description'])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Common Symptoms:**")
                                for symptom in info['symptoms']:
                                    st.write(f"- {symptom}")
                            
                            with col2:
                                st.markdown("**Treatment Options:**")
                                for treatment in info['treatment']:
                                    st.write(f"- {treatment}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with tab3:
        st.header("About DermaScan AI")
        st.write("""
        This skin disease classification app uses a deep learning model trained to identify 9 common skin conditions.
        
        **How it works:**
        - The model analyzes uploaded images of skin conditions
        - It provides predictions with confidence levels
        - You get instant information about the predicted condition
        
        **Supported Conditions:**
        """)
        
        # Display all conditions with icons
        cols = st.columns(3)
        for i, disease in enumerate(disease_info.keys()):
            with cols[i % 3]:
                st.markdown(f"✅ {disease_info[disease]['icon']} {disease}")
        
        st.markdown("""
        ---
        **Disclaimer:**  
        This app provides informational predictions only and is not a substitute for professional medical advice, 
        diagnosis, or treatment. Always consult a healthcare professional for medical concerns.
        """)

if __name__ == "__main__":
    main()