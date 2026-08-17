import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Penyakit Buah Tomat",
    page_icon="🍅",
    layout="centered"
)


# ============================================================
# KONFIGURASI MODEL
# ============================================================

MODEL_PATH = "best_model_70_15_15.keras"
IMG_SIZE = (224, 224)


# ============================================================
# LABEL KELAS
# Urutan harus sama dengan class_indices saat training
# ============================================================

CLASS_LABELS = [
    "Anthracnose (Colletotrichum spp.)",
    "Bacterial Spot (Xanthomonas spp.)",
    "Black Mold Rot (Alternaria alternata)",
    "Late Blight (Phytophthora infestans)"
]


# ============================================================
# INFORMASI PENYAKIT
# ============================================================

CLASS_INFO = {

    "Anthracnose (Colletotrichum spp.)":
        "Disebabkan oleh jamur Colletotrichum spp. "
        "Gejala berupa bercak melingkar cekung berwarna "
        "cokelat kehitaman pada permukaan buah.",

    "Bacterial Spot (Xanthomonas spp.)":
        "Disebabkan oleh bakteri Xanthomonas spp. "
        "Gejala berupa bercak kecil berair yang kemudian "
        "mengeras dan menjadi bercak cokelat kasar.",

    "Black Mold Rot (Alternaria alternata)":
        "Disebabkan oleh jamur Alternaria alternata. "
        "Gejala berupa busuk berwarna hitam kehijauan "
        "dengan tekstur lunak pada bagian buah.",

    "Late Blight (Phytophthora infestans)":
        "Disebabkan oleh Phytophthora infestans. "
        "Gejala berupa bercak cokelat keabuan yang "
        "basah dan dapat menyebar pada permukaan buah."
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_classification_model():
    return load_model(MODEL_PATH)


model = load_classification_model()


# ============================================================
# ANTARMUKA
# ============================================================

st.title("🍅 Klasifikasi Penyakit pada Buah Tomat")

st.write(
    "Unggah citra buah tomat untuk mengetahui jenis penyakitnya "
    "menggunakan model Convolutional Neural Network (CNN) "
    "dengan arsitektur MobileNetV2."
)


# ============================================================
# UPLOAD GAMBAR
# ============================================================

uploaded_file = st.file_uploader(
    "Unggah citra buah tomat",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROSES PREDIKSI
# ============================================================

if uploaded_file is not None:

    # Membaca gambar
    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Citra yang diunggah",
        width=500
    )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    img_resized = img.resize(IMG_SIZE)

    img_array = image.img_to_array(img_resized)

    # Normalisasi sesuai proses training
    img_array = img_array / 255.0

    # Menambahkan batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # --------------------------------------------------------
    # PREDIKSI
    # --------------------------------------------------------

    with st.spinner("Menganalisis citra..."):

        predictions = model.predict(
            img_array,
            verbose=0
        )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    predicted_class = CLASS_LABELS[predicted_index]

    confidence = float(
        predictions[0][predicted_index]
    ) * 100

    # --------------------------------------------------------
    # HASIL PREDIKSI
    # --------------------------------------------------------

    st.success(
        f"Hasil Deteksi: {predicted_class}"
    )

    st.metric(
        "Confidence Score",
        f"{confidence:.2f}%"
    )

    st.info(
        CLASS_INFO[predicted_class]
    )

    # --------------------------------------------------------
    # PROBABILITAS SETIAP KELAS
    # --------------------------------------------------------

    st.subheader("Probabilitas Setiap Kelas")

    for i, label in enumerate(CLASS_LABELS):

        probability = float(
            predictions[0][i]
        ) * 100

        st.write(
            f"**{label}**: {probability:.2f}%"
        )


else:

    st.info(
        "Silakan unggah citra buah tomat terlebih dahulu."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Model: MobileNetV2 (Transfer Learning) | "
    "Kelas: Anthracnose, Bacterial Spot, "
    "Black Mold Rot, Late Blight"
)
