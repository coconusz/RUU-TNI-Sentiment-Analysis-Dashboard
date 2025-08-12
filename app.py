import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix

# Load data
df = pd.read_csv("ruutni_cleaned_final.csv")
df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)
df['created_at_date'] = df['created_at'].dt.date

# Tema warna biru gelap
palette_biru_gelap = {
    "positif": "#799EFF",
    "negatif": "#EA5B6F"
}

# Fungsi untuk variasi warna wordcloud
color_map = {
    "positif": ["#799EFF", "#4D7CFE", "#A7C8FF", "#1A5DFF"],
    "negatif": ["#EA5B6F", "#D9364F", "#FF7A8A", "#A82339"]
}

def random_color_func(sentiment):
    def color_func(*args, **kwargs):
        return random.choice(color_map[sentiment])
    return color_func

# Sidebar navigasi
st.sidebar.title("🔍 Navigasi Aplikasi")
menu = st.sidebar.radio("Pilih Halaman", ["📊 Dashboard", "📈 Evaluasi Model"])

if menu == "📊 Dashboard":
    st.sidebar.subheader("📅 Filter Waktu")
    min_date = df['created_at_date'].min()
    max_date = df['created_at_date'].max()
    date_range = st.sidebar.date_input("Pilih Rentang Waktu", [min_date, max_date], min_value=min_date, max_value=max_date)

    # Filter data
    df_filtered = df[(df['created_at_date'] >= date_range[0]) &
                 (df['created_at_date'] <= date_range[1])].copy()
    df_filtered['date'] = df_filtered['created_at'].dt.date

    # Layout dashboard
    st.markdown("""
        <div style='padding-left: 8vw; padding-right: 8vw;'>
    """, unsafe_allow_html=True)

    st.title("Sentiment Analysis Dashboard")
    st.write("Dashboard ini berisi analisis sentimen komentar masyarakat di Twitter terhadap RUU TNI.")

    # ===== Distribusi Sentimen & keyword sentiment =====
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df_filtered['label'].value_counts()
        fig1, ax1 = plt.subplots(facecolor='none')
        colors = [palette_biru_gelap.get(label.lower(), "gray") for label in sentiment_counts.index]
        wedges, texts, autotexts = ax1.pie(
            sentiment_counts, labels=None, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops=dict(color="white")
        )
        for i, text in enumerate(sentiment_counts.index):
            autotexts[i].set_text(f"{text.capitalize()}\n{autotexts[i].get_text()}")
        for text in texts:
            text.set_color("white")
        ax1.axis('equal')
        fig1.patch.set_alpha(0.0)
        st.pyplot(fig1)

    with col2:
        st.subheader("Keyword Sentiment Distribution")
        from collections import Counter
        def clean_text(text):
            try:
                return eval(text)
            except:
                return []

        keyword_df = df_filtered.copy()
        keyword_df['tokens'] = keyword_df['full_text'].apply(clean_text)
        exploded = keyword_df.explode('tokens')
        keyword_counts = exploded['tokens'].value_counts().head(10)
        fig_kw, ax_kw = plt.subplots(facecolor='none')
        ax_kw.set_facecolor('none')
        sns.barplot(x=keyword_counts.values, y=keyword_counts.index, ax=ax_kw, color='#3399ff')
        ax_kw.tick_params(colors='white')
        ax_kw.spines['bottom'].set_color('white')
        ax_kw.spines['left'].set_color('white')
        ax_kw.yaxis.label.set_color('white')
        ax_kw.xaxis.label.set_color('white')
        ax_kw.title.set_color('white')
        for label in ax_kw.get_xticklabels():
            label.set_color("white")
        for label in ax_kw.get_yticklabels():
            label.set_color("white")
        fig_kw.patch.set_alpha(0.0)
        st.pyplot(fig_kw)

    # ===== Tren Sentimen =====
    st.subheader("Tren Sentimen dari Waktu ke Waktu")
    trend = df_filtered.groupby(['date', 'label']).size().unstack().fillna(0)
    fig_trend, ax_trend = plt.subplots(facecolor='none')
    ax_trend.set_facecolor('none')
    trend.plot(ax=ax_trend,
               color=[palette_biru_gelap.get(col.lower(), "gray") for col in trend.columns])
    ax_trend.tick_params(colors='white')
    ax_trend.spines['bottom'].set_color('white')
    ax_trend.spines['left'].set_color('white')
    ax_trend.yaxis.label.set_color('white')
    ax_trend.xaxis.label.set_color('white')
    ax_trend.title.set_color('white')
    for label in ax_trend.get_xticklabels():
        label.set_color("white")
    for label in ax_trend.get_yticklabels():
        label.set_color("white")
    fig_trend.patch.set_alpha(0.0)
    st.pyplot(fig_trend)

    # ===== Word Cloud =====
    st.subheader("Word Cloud")
    for sentiment in ['positif', 'negatif']:
        st.markdown(f"**Sentimen: {sentiment.capitalize()}**")
        all_words = df[df['label'].str.lower() == sentiment]['full_text'].apply(clean_text).explode()
        word_freq = all_words.value_counts().to_dict()
        wordcloud = WordCloud(width=800, height=400, background_color=None, mode="RGBA",
                              color_func=random_color_func(sentiment)).generate_from_frequencies(word_freq)
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_alpha(0.0)
        st.pyplot(fig)

    # ===== Tabel Data Interaktif =====
    st.subheader("Tabel Data Interaktif")
    st.dataframe(df_filtered[['created_at', 'full_text', 'label']])

    st.markdown("""</div>""", unsafe_allow_html=True)

elif menu == "📈 Evaluasi Model":
    st.title("Evaluasi Performa Model Sentimen")
    st.write("Berikut adalah confusion matrix dari model klasifikasi sentimen:")

    # Tampilkan gambar confusion matrix yang sudah ada
    st.subheader("Confusion Matrix")
    st.image("confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)

    st.subheader("Interpretasi Confusion Matrix")
    st.markdown("""
    - **True Positive (Positif - Positif)**: Model berhasil memprediksi sentimen positif dengan benar.
    - **True Negative (Negatif - Negatif)**: Model berhasil memprediksi sentimen negatif dengan benar.
    - **False Positive (Negatif → Positif)**: Model salah memprediksi negatif sebagai positif.
    - **False Negative (Positif → Negatif)**: Model salah memprediksi positif sebagai negatif.
    """)

# Nilai dari confusion matrix
    TP = 101
    FN = 31
    FP = 12
    TN = 307

    # Hitung metrik evaluasi
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    st.subheader("📊 Skor Evaluasi Model")
    eval_df = pd.DataFrame({
        "Metrik": ["Akurasi", "Presisi", "Recall", "F1-Score"],
        "Skor": [accuracy, precision, recall, f1_score]
    })
    eval_df["Skor"] = eval_df["Skor"].apply(lambda x: round(x, 3))

    st.table(eval_df)
