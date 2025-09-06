# app.py
import streamlit as st
import pandas as pd
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# CONFIG
MODEL_NAME = "aisyahnoviani16/miningComments"
id2label = {
    0: "😢 KESEDIHAN",
    1: "😡 AMARAH",
    2: "🤝 DUKUNGAN",
    3: "🌟 HARAPAN",
    4: "😞 KEKECEWAAN"
}

st.set_page_config(page_title="📝 Emotion Mining App", page_icon="📝", layout="centered")

# Custom CSS untuk desain
st.markdown("""
<style>
/* Global App Background (putih polos) */
.stApp {
    background-color: #fff;
    font-family: "Segoe UI", sans-serif;
}

/*  Kontainer Utama dengan card dan gradient pink  */
.block-container {
    background: linear-gradient(135deg, #ffe6f2, #ffd9ec);
    border-radius: 15px;
    padding-top: 2rem;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
}

/* Judul */
h1, h2, h3 {
    color: #a83279;
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #fdfbfb, #f5d0eb);
}
section[data-testid="stSidebar"] .css-1d391kg { 
    background: transparent; 
}

/* Tombol Prediksi */
.stButton>button {
    background: #ff4d94;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: 0.3s;
}
.stButton>button:hover {
    background: #e60073;
    color: white;
}

/* Input Area */
textarea {
    border-radius: 10px !important;
    border: 1px solid #ffb3d9 !important;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
@st.cache_resource
def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

nlp = load_pipeline()

# HEADER
st.title("📝 Emotion Mining App")
st.markdown("""
Aplikasi ini menganalisis komentar  
dan mengklasifikasikan ke dalam **5 emosi utama**:

- 😢 **KESEDIHAN**  
- 😡 **AMARAH**  
- 🤝 **DUKUNGAN**  
- 🌟 **HARAPAN**  
- 😞 **KEKECEWAAN**  
""")

# SIDEBAR - Instruksi
st.sidebar.header("📌 Instruksi")
st.sidebar.info("""
1. Pilih mode input (ketik, batch teks, atau upload CSV).  
2. Masukkan komentar atau file.  
3. Tekan tombol **Prediksi**.  
4. Lihat hasil & unduh jika perlu.  
""")

mode = st.sidebar.radio("Pilih Mode Input", ["Ketik Komentar", "Batch Teks", "Upload CSV"])

# MODE 1: Ketik Komentar
if mode == "Ketik Komentar":
    user_input = st.text_area(" Masukkan komentar :",
                               placeholder="Contoh: Saya kecewa dengan keputusan pengadilan ini...")
    if st.button("🔍 Prediksi Emosi"):
        if user_input.strip():
            with st.spinner("Sedang memproses..."):
                results = nlp(user_input)[0]

            labels = [id2label[int(r["label"].split("_")[-1])] for r in results]
            scores = [r["score"] for r in results]

            # Prediksi utama
            top_idx = scores.index(max(scores))
            st.markdown(f"<div class='emotion-badge'>Prediksi Utama: {labels[top_idx]} "
                        f"(Confidence: {scores[top_idx]:.2f})</div>", unsafe_allow_html=True)

            # Grafik confidence
            df = pd.DataFrame({"Emosi": labels, "Confidence": scores})
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Emosi", sort="-y"),
                y="Confidence",
                color="Emosi"
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("⚠️ Tolong masukkan komentar dulu.")

# MODE 2: Batch Input (multi-line)
elif mode == "Batch Teks":
    batch_input = st.text_area(" Masukkan beberapa komentar (pisahkan dengan baris baru):",
                               placeholder="contoh:\nSaya kecewa sekali\nSaya mendukung penuh keputusan ini")
    if st.button("🚀 Prediksi "):
        texts = [t.strip() for t in batch_input.split("\n") if t.strip()]
        if texts:
            with st.spinner("Sedang memproses..."):
                preds = []
                for txt in texts:
                    result = nlp(txt)[0]
                    top = max(result, key=lambda r: r["score"])
                    preds.append(id2label[int(top["label"].split("_")[-1])])

                df = pd.DataFrame({"Teks": texts, "Prediksi": preds})
                st.success("✅ Prediksi batch selesai!")
                st.dataframe(df)

                st.download_button("💾 Download Hasil CSV",
                                   df.to_csv(index=False).encode("utf-8"),
                                   "batch_predictions.csv", "text/csv")
        else:
            st.warning("⚠️ Masukkan minimal 1 komentar.")

# MODE 3: Upload CSV
elif mode == "Upload CSV":
    st.info("Format CSV harus ada kolom: **id, text**.")
    uploaded = st.file_uploader("📂 Upload file CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)

        # ✅ Pastikan kolom text ada
        if "text" not in df.columns:
            st.error("❌ CSV harus punya kolom 'text'")
        else:
            # ✅ Bersihkan data kosong
            df = df[df["text"].notna()]              # buang NaN
            df = df[df["text"].str.strip() != ""]    # buang baris kosong
            df = df.reset_index(drop=True)

            if df.empty:
                st.warning("⚠️ Tidak ada teks valid di file CSV.")
            else:
                preds = []
                error_rows = []

                with st.spinner("Sedang memproses..."):
                    for i, txt in enumerate(df["text"].astype(str)):
                        try:
                            result = nlp(txt)[0]
                            top = max(result, key=lambda r: r["score"])
                            preds.append(id2label[int(top["label"].replace("LABEL_", ""))])
                        except Exception as e:
                            preds.append("⚠️ ERROR")
                            error_rows.append(i)
                            # Kalau mau log error, bisa tulis:
                            # st.write(f"Error di baris {i}: {e}")

                df["Predicted_Emotion"] = preds

                if error_rows:
                    st.warning(f"⚠️ {len(error_rows)} baris gagal diproses (kosong/invalid).")

                st.success("✅ Prediksi selesai!")
                st.dataframe(df.head(10))

                st.download_button("💾 Download Hasil CSV",
                                   df.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")


