import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Penyakit Buah Tomat",
    page_icon="🍅",
    layout="centered",
)

MODEL_PATH = "best_model_70_15_15.keras"
IMG_SIZE = (224, 224)

# Urutan kelas HARUS sama persis dengan urutan hasil flow_from_directory()
# saat training (class_indices), yaitu urutan alfabetis nama folder dataset.
CLASS_LABELS = [
    "Anthracnose (Colletotrichum spp.)",
    "Bacterial Spot (Xanthomonas spp.)",
    "Black Mold Rot (Alternaria alternata)",
    "Late Blight (Phytophthora infestans)",
]

# Deskripsi singkat tiap penyakit untuk ditampilkan ke pengguna
CLASS_INFO = {
    "Anthracnose (Colletotrichum spp.)": (
        "Disebabkan oleh jamur *Colletotrichum spp.* Gejala berupa bercak "
        "melingkar cekung berwarna cokelat kehitaman pada permukaan buah."
    ),
    "Bacterial Spot (Xanthomonas spp.)": (
        "Disebabkan oleh bakteri *Xanthomonas spp.* Gejala berupa bercak "
        "kecil berair yang mengeras dan menjadi bercak cokelat kasar pada kulit buah."
    ),
    "Black Mold Rot (Alternaria alternata)": (
        "Disebabkan oleh jamur *Alternaria alternata*. Gejala berupa busuk "
        "hitam kehijauan dengan tekstur lunak pada bagian buah yang terinfeksi."
    ),
    "Late Blight (Phytophthora infestans)": (
        "Disebabkan oleh jamur *Phytophthora infestans*. Gejala berupa "
        "bercak cokelat keabuan basah yang menyebar cepat pada permukaan buah."
    ),
}


# ----------------------------------------------------------------------------
# LOAD MODEL (di-cache agar tidak dimuat ulang setiap interaksi pengguna)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_classification_model():
    return load_model(MODEL_PATH)


model = load_classification_model()

# ----------------------------------------------------------------------------
# ANTARMUKA APLIKASI
# ----------------------------------------------------------------------------
st.title("🍅 Deteksi Penyakit pada Buah Tomat")
st.write(
    "Unggah citra buah tomat untuk mendeteksi jenis penyakit menggunakan "
    "model *Convolutional Neural Network* (CNN) dengan arsitektur MobileNetV2."
)

uploaded_file = st.file_uploader(
    "Unggah citra buah tomat (format JPG/JPEG/PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    # Tampilkan citra yang diunggah
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Citra yang diunggah", use_container_width=True)

    # ------------------------------------------------------------------
    # PRAPROSES CITRA (harus sama persis dengan praproses saat training:
    # resize 224x224 dan normalisasi rescale 1/255)
    # ------------------------------------------------------------------
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # bentuk (1, 224, 224, 3)

    # ------------------------------------------------------------------
    # PREDIKSI
    # ------------------------------------------------------------------
    with st.spinner("Menganalisis citra..."):
        predictions = model.predict(img_array)

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_LABELS[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    # ------------------------------------------------------------------
    # TAMPILKAN HASIL
    # ------------------------------------------------------------------
    st.success(f"**Hasil Deteksi:** {predicted_class}")
    st.write(f"**Tingkat Keyakinan (Confidence Score):** {confidence:.2f}%")
    st.info(CLASS_INFO[predicted_class])

    # Tampilkan probabilitas untuk seluruh kelas
    st.subheader("Probabilitas Tiap Kelas")
    for label, prob in zip(CLASS_LABELS, predictions[0]):
        st.write(f"{label}: {prob * 100:.2f}%")
        st.progress(float(prob))
else:
    st.warning("Silakan unggah citra buah tomat terlebih dahulu.")

st.markdown("---")
st.caption(
    "Model: MobileNetV2 (Transfer Learning) | "
    "Kelas: Anthracnose, Bacterial Spot, Black Mold Rot, Late Blight"
)
