import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO

st.set_page_config(
    page_title="Apple vs Orange Classifier",
    layout="wide"
)

IMG_SIZE = (160, 160)

# ==============================
# Custom CSS
# ==============================
st.markdown("""
<style>
h1 { text-align: center; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

class_names = ['Apple', 'Orange']

model_paths = {
    "Custom CNN": "custom_cnn_apple_orange.h5",
    "Transfer Learning (MobileNetV2)": "mobilenetv2_apple_orange.h5"
}

with st.sidebar:
    st.header("Pengaturan")
    model_choice = st.selectbox("Pilih Model", list(model_paths.keys()))
    st.write("---")
    st.header("Tentang Model")
    st.write(f"Model aktif: **{model_choice}**")
    st.write(f"Ukuran input: {IMG_SIZE}")

model = load_model(model_paths[model_choice])

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["🔍 Prediksi", "📊 Tentang Model"])

with tab1:
    st.title("Klasifikasi Apple vs Orange")
    st.write(f"Menggunakan model: **{model_choice}**")

    uploaded_file = st.file_uploader(
    "Pilih gambar...", 
    type=['png', 'jpg', 'jpeg'],
    key=f"uploader_{model_choice}"
)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_resized = image.resize(IMG_SIZE)

        img_array = np.array(image_resized).astype('float32') / 255.0
        img_array = img_array.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3)

        with st.spinner('Menganalisis gambar...'):
            prediction = model.predict(img_array)

        raw_score = prediction[0][0]
        predicted_class = class_names[int(raw_score > 0.5)]
        confidence = raw_score if raw_score > 0.5 else 1 - raw_score
        prob_orange = raw_score
        prob_apple = 1 - raw_score

        emoji = "🍎" if predicted_class == "Apple" else "🍊"

        col_left, col_right = st.columns([1, 2])

        with col_left:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            st.markdown(f"""
            <div style="width: 100%; height: 220px; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <img src="data:image/png;base64,{img_base64}" style="width: 100%; height: 100%; object-fit: cover;">
            </div>
            <p style="text-align: center; color: #666; font-size: 13px; margin-top: 6px;">Gambar diupload</p>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown(f"""
            <div style="background-color: var(--secondary-background-color, #F8F9FA); border-radius: 12px; padding: 20px; border: 1px solid #E0E0E0; height: 220px; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;">
                <p style="color: #666; margin: 0; font-size: 18px;">Prediksi</p>
                <h2 style="margin: 8px 0; font-size: 36px;">{predicted_class} {emoji}</h2>
                <p style="color: #666; margin: 0; font-size: 18px;">Confidence: {confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        if confidence > 0.8:
            st.success(f"Model cukup yakin: **{predicted_class}** ({confidence*100:.1f}%)")
            st.progress(float(confidence))
        elif confidence > 0.5:
            st.warning(f"Model kurang yakin: **{predicted_class}** ({confidence*100:.1f}%)")
            st.progress(float(confidence))
        
        st.write("### Perbandingan Probabilitas")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(label="🍎 Apple", value=f"{prob_apple*100:.1f}%")
        with col_b:
            st.metric(label="🍊 Orange", value=f"{prob_orange*100:.1f}%")

        st.session_state.history.append({
            "Model": model_choice,
            "File": uploaded_file.name,
            "Prediksi": predicted_class,
            "Confidence": f"{confidence*100:.1f}%"
        })

    if st.session_state.history:
        st.write("### Riwayat Prediksi")
        st.table(st.session_state.history)
    
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()

with tab2:
    st.header("Tentang Model")
    st.write(f"**Model aktif:** {model_choice}")
    st.write(f"**Ukuran input:** {IMG_SIZE}")
    st.write("**Kelas:** Apple, Orange")
    st.write("---")
    if model_choice == "Custom CNN":
        st.write("Model CNN yang dibangun dari nol, terdiri dari 4 layer Conv2D + MaxPooling2D.")
    else:
        st.write("Model menggunakan transfer learning MobileNetV2.")