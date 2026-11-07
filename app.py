
import os
import io
import json
import zipfile
import warnings
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ==========================================================
# 1. KONFIGURASI
# ==========================================================

st.set_page_config(
    page_title="K-Means Clustering Gempa Bumi",
    page_icon="🌍",
    layout="wide"
)

RANDOM_STATE = 42
OUTPUT_DIR = Path("hasil")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "latitude",
    "longitude",
    "depth",
    "mag",
    "tsunami",
    "sig",
    "nst",
    "dmin",
    "rms",
    "gap",
    "confidence_score"
]

st.title("🌍 Analisis Clustering Data Gempa Bumi")
st.markdown(
    """
    Dashboard ini menerapkan algoritma **K-Means Clustering**
    untuk mengelompokkan data gempa bumi berdasarkan karakteristik
    lokasi, kedalaman, magnitudo, dan fitur numerik lainnya.
    """
)

# ==========================================================
# 2. FUNGSI MEMBACA DATASET
# ==========================================================

def read_zip_file(file_object):
    with zipfile.ZipFile(file_object, "r") as zf:

        csv_files = [
            name for name in zf.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError(
                "Tidak ditemukan file CSV di dalam ZIP."
            )

        selected_csv = next(
            (
                name for name in csv_files
                if "global_earthquakes_seismic"
                in Path(name).name.lower()
            ),
            csv_files[0]
        )

        with zf.open(selected_csv) as file:
            data = pd.read_csv(file)

    return data, selected_csv


def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))

    if file_name.endswith(".zip"):
        data, _ = read_zip_file(io.BytesIO(file_bytes))
        return data

    raise ValueError("Format file harus CSV atau ZIP.")


def read_local_dataset():

    possible_csv = [
        Path("archive (1)/global_earthquakes_seismic.csv"),
        Path("global_earthquakes_seismic.csv"),
        Path("dataset/global_earthquakes_seismic.csv")
    ]

    for csv_path in possible_csv:
        if csv_path.exists():
            return pd.read_csv(csv_path)

    possible_zip = [
        Path("dataset/archive (1).zip"),
        Path("archive (1).zip")
    ]

    for zip_path in possible_zip:
        if zip_path.exists():
            data, _ = read_zip_file(zip_path)
            return data

    # Mencari file ZIP lain jika nama berbeda
    candidates = []

    for folder in [Path("dataset"), Path(".")]:
        if folder.exists():
            candidates.extend(folder.glob("*.zip"))

    for zip_path in candidates:
        try:
            data, _ = read_zip_file(zip_path)
            return data
        except Exception:
            continue

    return None


# ==========================================================
# 3. UPLOAD DATASET
# ==========================================================

st.sidebar.header("Pengaturan Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset CSV atau ZIP",
    type=["csv", "zip"]
)

try:
    if uploaded_file is not None:
        df = read_uploaded_file(uploaded_file)
    else:
        df = read_local_dataset()

except Exception as error:
    st.error(f"Gagal membaca dataset: {error}")
    st.stop()


if df is None:
    st.warning("Dataset belum ditemukan.")

    st.info(
        "Silakan upload file dataset melalui sidebar "
        "atau letakkan dataset pada folder yang sesuai."
    )

    st.code(
        "Praktikum_KMeans/\n"
        "├── app.py\n"
        "├── archive (1)/\n"
        "│   └── global_earthquakes_seismic.csv\n"
        "└── hasil/"
    )

    st.stop()


if df.empty:
    st.error("Dataset kosong.")
    st.stop()


# ==========================================================
# 4. INFORMASI DATASET
# ==========================================================

st.header("1. Informasi Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Jumlah Baris", f"{df.shape[0]:,}")
col2.metric("Jumlah Kolom", df.shape[1])
col3.metric(
    "Jumlah Duplikat",
    f"{df.duplicated().sum():,}"
)

st.subheader("Data Awal")

st.dataframe(
    df.head(),
    use_container_width=True
)

with st.expander("Informasi Struktur Dataset"):

    st.write("Nama Kolom:")
    st.write(df.columns.tolist())

    info_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str).values,
        "Jumlah Missing": df.isnull().sum().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

with st.expander("Statistik Deskriptif"):

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )


# ==========================================================
# 5. PEMILIHAN FITUR
# ==========================================================

st.header("2. Pemilihan Fitur Clustering")

missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    st.error(
        "Kolom fitur berikut tidak ditemukan: "
        + ", ".join(missing_features)
    )
    st.stop()

features = FEATURES.copy()

data = df[features].copy()

data = data.apply(
    pd.to_numeric,
    errors="coerce"
)

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

st.write("Fitur yang digunakan dalam clustering:")

st.write(features)

st.write("Ukuran data fitur:", data.shape)

with st.expander("Lihat Data Fitur"):

    st.dataframe(
        data.head(),
        use_container_width=True
    )


# ==========================================================
# 6. MISSING VALUE DAN DUPLIKAT
# ==========================================================

st.header("3. Pemeriksaan Missing Value dan Duplikat")

missing_table = data.isna().sum().to_frame(
    "Jumlah Missing Value"
)

col1, col2 = st.columns(2)

col1.metric(
    "Total Missing Value",
    int(data.isna().sum().sum())
)

col2.metric(
    "Jumlah Baris Duplikat",
    int(df.duplicated().sum())
)

st.subheader("Missing Value Setiap Fitur")

st.dataframe(
    missing_table,
    use_container_width=True
)


# ==========================================================
# 7. EXPLORATORY DATA ANALYSIS
# ==========================================================

st.header("4. Exploratory Data Analysis (EDA)")

eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
    "Histogram",
    "Boxplot",
    "Heatmap Korelasi",
    "Pairplot"
])


# ---------------- HISTOGRAM ----------------

with eda_tab1:

    st.subheader("Distribusi Fitur Numerik")

    fig, ax = plt.subplots(figsize=(14, 9))

    data.hist(
        bins=30,
        edgecolor="black",
        ax=None,
        figsize=(14, 9),
        color="#56A6D9"
    )

    st.pyplot(plt.gcf())
    plt.close("all")

    st.write(
        "Histogram digunakan untuk melihat distribusi "
        "dan frekuensi nilai pada setiap fitur numerik."
    )


# ---------------- BOXPLOT ----------------

with eda_tab2:

    st.subheader("Boxplot Fitur Numerik")

    fig, ax = plt.subplots(figsize=(15, 7))

    sns.boxplot(
        data=data,
        ax=ax,
        color="#56A6D9"
    )

    ax.set_title("Boxplot Fitur Numerik Gempa Bumi")
    ax.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "Boxplot digunakan untuk melihat persebaran data "
        "dan kemungkinan adanya pencilan atau outlier."
    )


# ---------------- HEATMAP ----------------

with eda_tab3:

    st.subheader("Correlation Heatmap")

    corr = data.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(13, 9))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.4,
        ax=ax
    )

    ax.set_title("Correlation Heatmap Fitur Gempa Bumi")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "Heatmap menunjukkan hubungan korelasi "
        "antarfitur numerik."
    )


# ---------------- PAIRPLOT ----------------

with eda_tab4:

    st.subheader("Pairplot Fitur Gempa Bumi")

    pair_features = [
        "latitude",
        "longitude",
        "depth",
        "mag"
    ]

    pair_data = data[pair_features].dropna()

    if len(pair_data) > 0:

        pair_sample = pair_data.sample(
            n=min(1000, len(pair_data)),
            random_state=RANDOM_STATE
        )

        pair_fig = sns.pairplot(
            pair_sample,
            diag_kind="hist",
            corner=True,
            plot_kws={"color": "#264A73"},
            diag_kws={"color": "#56A6D9"}
        )

        pair_fig.fig.suptitle(
            "Pairplot Lokasi, Kedalaman, dan Magnitudo",
            y=1.02
        )

        st.pyplot(pair_fig.fig)
        plt.close("all")

        st.write(
            "Pairplot memperlihatkan hubungan antarfitur "
            "latitude, longitude, depth, dan magnitudo."
        )

    else:
        st.warning("Data pairplot tidak tersedia.")


# ==========================================================
# 8. PREPROCESSING DATA
# ==========================================================

st.header("5. Preprocessing Data")

# Menghapus duplikasi seluruh kolom dataset
df_clean = df.drop_duplicates().reset_index(drop=True)

# Mengambil fitur dari dataset bersih
X = df_clean[features].copy()

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Mengisi missing value menggunakan median
medians = X.median()

# Memeriksa kolom yang seluruhnya kosong
if medians.isna().any():

    bad_cols = medians[
        medians.isna()
    ].index.tolist()

    st.error(
        "Fitur berikut seluruh nilainya kosong atau tidak valid: "
        + ", ".join(bad_cols)
    )

    st.stop()

X = X.fillna(medians)

if X.isna().any().any():
    st.error("Masih terdapat missing value setelah imputasi.")
    st.stop()

if len(X) < 10:
    st.error("Data harus memiliki minimal 10 baris untuk proses ini.")
    st.stop()

# Standardisasi
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=features,
    index=X.index
)

st.success("Preprocessing berhasil dilakukan.")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Data Setelah Hapus Duplikat",
    f"{len(df_clean):,}"
)

col2.metric(
    "Jumlah Fitur",
    len(features)
)

col3.metric(
    "Missing Value Setelah Imputasi",
    int(X.isna().sum().sum())
)

with st.expander("Data Sebelum Standardisasi"):

    st.dataframe(
        X.head(),
        use_container_width=True
    )

with st.expander("Data Setelah Standardisasi"):

    st.dataframe(
        X_scaled_df.head(),
        use_container_width=True
    )

# Simpan hasil preprocessing
X.to_csv(
    OUTPUT_DIR / "data_sebelum_standarisasi.csv",
    index=False
)

X_scaled_df.to_csv(
    OUTPUT_DIR / "data_preprocessing.csv",
    index=False
)

joblib.dump(
    scaler,
    OUTPUT_DIR / "standard_scaler.joblib"
)

with open(
    OUTPUT_DIR / "fitur_clustering.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        features,
        file,
        ensure_ascii=False,
        indent=2
    )


# ==========================================================
# 9. ELBOW METHOD
# ==========================================================

st.header("6. Menentukan Jumlah Cluster - Elbow Method")

k_values = list(range(2, 11))

inertias = []

progress = st.progress(0)

for i, k in enumerate(k_values):

    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    km.fit(X_scaled)

    inertias.append(km.inertia_)

    progress.progress(
        (i + 1) / len(k_values)
    )

progress.empty()

elbow_df = pd.DataFrame({
    "K": k_values,
    "Inertia": inertias
})

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    k_values,
    inertias,
    marker="o",
    color="#264A73"
)

ax.set_xticks(k_values)
ax.set_xlabel("Jumlah Cluster (K)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method")
ax.grid(True)

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    elbow_df,
    use_container_width=True
)

elbow_df.to_csv(
    OUTPUT_DIR / "hasil_elbow.csv",
    index=False
)

st.write(
    "Elbow Method digunakan untuk melihat penurunan nilai inertia "
    "ketika jumlah cluster bertambah. Titik siku dapat dijadikan "
    "pertimbangan untuk memilih jumlah cluster."
)


# ==========================================================
# 10. SILHOUETTE SCORE
# ==========================================================

st.header("7. Evaluasi Silhouette Score")

silhouette_scores = []

sil_sample_size = min(
    3000,
    len(X_scaled)
)

progress = st.progress(0)

for i, k in enumerate(k_values):

    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels_k = km.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels_k,
        sample_size=sil_sample_size,
        random_state=RANDOM_STATE
    )

    silhouette_scores.append(score)

    progress.progress(
        (i + 1) / len(k_values)
    )

progress.empty()

silhouette_df = pd.DataFrame({
    "K": k_values,
    "Silhouette Score": silhouette_scores
})

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    k_values,
    silhouette_scores,
    marker="o",
    color="#56A6D9"
)

ax.set_xticks(k_values)
ax.set_xlabel("Jumlah Cluster (K)")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Score untuk Setiap K")
ax.grid(True)

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    silhouette_df,
    use_container_width=True
)

silhouette_df.to_csv(
    OUTPUT_DIR / "hasil_silhouette.csv",
    index=False
)

best_k_silhouette = int(
    silhouette_df.loc[
        silhouette_df["Silhouette Score"].idxmax(),
        "K"
    ]
)

st.success(
    f"K dengan Silhouette Score tertinggi: {best_k_silhouette}"
)

st.write(
    "Silhouette Score digunakan untuk mengukur kualitas "
    "pemisahan antarcluster. Nilai yang lebih tinggi "
    "menunjukkan pemisahan cluster yang lebih baik."
)


# ==========================================================
# 11. PROSES K-MEANS CLUSTERING
# ==========================================================

st.header("8. Proses K-Means Clustering")

# Default mengikuti notebook: K dengan Silhouette tertinggi
K_FINAL = st.sidebar.selectbox(
    "Pilih Jumlah Cluster Akhir",
    options=k_values,
    index=k_values.index(best_k_silhouette)
)

st.write(
    f"Jumlah cluster yang digunakan: **{K_FINAL}**"
)

kmeans = KMeans(
    n_clusters=K_FINAL,
    random_state=RANDOM_STATE,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

hasil = df_clean.copy()

hasil["cluster"] = cluster_labels

st.success("Proses K-Means berhasil dilakukan.")

st.dataframe(
    hasil.head(),
    use_container_width=True
)


# ==========================================================
# 12. EVALUASI MODEL
# ==========================================================

st.header("9. Hasil Evaluasi K-Means")

final_inertia = float(
    kmeans.inertia_
)

final_silhouette = float(
    silhouette_score(
        X_scaled,
        cluster_labels,
        sample_size=min(3000, len(X_scaled)),
        random_state=RANDOM_STATE
    )
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Jumlah Cluster",
    K_FINAL
)

col2.metric(
    "Inertia",
    f"{final_inertia:.4f}"
)

col3.metric(
    "Silhouette Score",
    f"{final_silhouette:.4f}"
)

evaluasi = pd.DataFrame({
    "Metrik": [
        "Jumlah Cluster",
        "Inertia",
        "Silhouette Score"
    ],
    "Nilai": [
        K_FINAL,
        final_inertia,
        final_silhouette
    ]
})

st.dataframe(
    evaluasi,
    use_container_width=True
)

evaluasi.to_csv(
    OUTPUT_DIR / "evaluasi_kmeans.csv",
    index=False
)


# ==========================================================
# 13. KARAKTERISTIK CLUSTER
# ==========================================================

st.header("10. Karakteristik Setiap Cluster")

cluster_mean = (
    hasil.groupby("cluster")[features]
    .mean()
    .round(3)
)

cluster_median = (
    hasil.groupby("cluster")[features]
    .median()
    .round(3)
)

cluster_count = (
    hasil["cluster"]
    .value_counts()
    .sort_index()
    .rename("jumlah_data")
)

tab1, tab2, tab3 = st.tabs([
    "Rata-rata",
    "Median",
    "Jumlah Anggota"
])

with tab1:

    st.subheader("Rata-rata Fitur per Cluster")

    st.dataframe(
        cluster_mean,
        use_container_width=True
    )

with tab2:

    st.subheader("Median Fitur per Cluster")

    st.dataframe(
        cluster_median,
        use_container_width=True
    )

with tab3:

    st.subheader("Jumlah Anggota Cluster")

    st.dataframe(
        cluster_count.to_frame(),
        use_container_width=True
    )

    count_df = cluster_count.reset_index()
    count_df.columns = ["Cluster", "Jumlah Data"]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        count_df["Cluster"].astype(str),
        count_df["Jumlah Data"],
        color="#264A73"
    )

    ax.set_title("Jumlah Data Setiap Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah Data")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ==========================================================
# 14. GRAFIK KARAKTERISTIK CLUSTER
# ==========================================================

st.header("11. Grafik Karakteristik Cluster")

summary_plot = cluster_mean[
    ["mag", "depth", "sig"]
]

fig, ax = plt.subplots(figsize=(11, 6))

summary_plot.plot(
    kind="bar",
    ax=ax,
    color=["#264A73", "#56A6D9", "#F39C3D"]
)

ax.set_title(
    "Rata-rata Magnitudo, Kedalaman, dan Signifikansi per Cluster"
)

ax.set_xlabel("Cluster")
ax.set_ylabel("Nilai Rata-rata")
ax.tick_params(
    axis="x",
    rotation=0
)

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.write(
    "Grafik ini membandingkan nilai rata-rata magnitudo, "
    "kedalaman, dan signifikansi pada setiap cluster."
)


# ==========================================================
# 15. SCATTER PLOT LOKASI GEMPA
# ==========================================================

st.header("12. Persebaran Lokasi Gempa Berdasarkan Cluster")

plot_df = hasil[
    ["latitude", "longitude", "cluster"]
].copy()

fig, ax = plt.subplots(figsize=(12, 7))

cluster_colors = [
    "#264A73",
    "#56A6D9",
    "#F39C3D",
    "#8E44AD",
    "#2ECC71",
    "#E74C3C",
    "#34495E",
    "#16A085",
    "#D35400",
    "#7F8C8D"
]

sns.scatterplot(
    data=plot_df,
    x="longitude",
    y="latitude",
    hue="cluster",
    palette=cluster_colors[:K_FINAL],
    s=22,
    alpha=0.65,
    ax=ax
)

ax.set_title(
    "Persebaran Lokasi Gempa Berdasarkan Cluster"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.legend(
    title="Cluster",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.write(
    "Scatter plot menunjukkan persebaran lokasi gempa bumi "
    "berdasarkan longitude dan latitude. Warna menunjukkan "
    "kelompok cluster."
)


# ==========================================================
# 16. VISUALISASI PCA
# ==========================================================

st.header("13. Visualisasi Cluster Menggunakan PCA")

pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["cluster"] = cluster_labels

fig, ax = plt.subplots(figsize=(10, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette=cluster_colors[:K_FINAL],
    alpha=0.7,
    s=25,
    ax=ax
)

ax.set_title("Visualisasi Cluster Menggunakan PCA")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend(title="Cluster")

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

explained = (
    pca.explained_variance_ratio_.sum() * 100
)

st.info(
    f"Dua komponen PCA menjelaskan {explained:.2f}% "
    "variasi data."
)


# ==========================================================
# 17. ANALISIS OTOMATIS CLUSTER
# ==========================================================

st.header("14. Analisis Hasil Clustering")

for cid in cluster_mean.index:

    n_data = int(
        cluster_count.loc[cid]
    )

    with st.expander(f"Analisis Cluster {cid}"):

        st.write(
            f"Jumlah anggota Cluster {cid}: **{n_data:,} data**."
        )

        st.write("Karakteristik rata-rata:")

        st.dataframe(
            cluster_mean.loc[[cid]],
            use_container_width=True
        )

        st.write("Karakteristik median:")

        st.dataframe(
            cluster_median.loc[[cid]],
            use_container_width=True
        )

        st.write(
            f"Rata-rata magnitudo: "
            f"**{cluster_mean.loc[cid, 'mag']:.2f}**"
        )

        st.write(
            f"Rata-rata kedalaman: "
            f"**{cluster_mean.loc[cid, 'depth']:.2f} km**"
        )

        st.write(
            f"Rata-rata signifikansi: "
            f"**{cluster_mean.loc[cid, 'sig']:.2f}**"
        )


# ==========================================================
# 18. SIMPAN HASIL CLUSTERING
# ==========================================================

st.header("15. Penyimpanan Hasil")

hasil.to_csv(
    OUTPUT_DIR / "hasil_clustering_gempa.csv",
    index=False
)

cluster_mean.to_csv(
    OUTPUT_DIR / "ringkasan_cluster_mean.csv"
)

cluster_median.to_csv(
    OUTPUT_DIR / "ringkasan_cluster_median.csv"
)

cluster_count.to_csv(
    OUTPUT_DIR / "jumlah_anggota_cluster.csv"
)

joblib.dump(
    kmeans,
    OUTPUT_DIR / "model_kmeans.joblib"
)

joblib.dump(
    pca,
    OUTPUT_DIR / "model_pca.joblib"
)

joblib.dump(
    scaler,
    OUTPUT_DIR / "standard_scaler.joblib"
)

# Simpan daftar fitur
with open(
    OUTPUT_DIR / "fitur_clustering.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        features,
        file,
        ensure_ascii=False,
        indent=2
    )


# ==========================================================
# 19. DOWNLOAD HASIL
# ==========================================================

st.header("16. Download Hasil Clustering")

csv_hasil = hasil.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Hasil Clustering CSV",
    data=csv_hasil,
    file_name="hasil_clustering_gempa.csv",
    mime="text/csv"
)

csv_mean = cluster_mean.to_csv().encode("utf-8")

st.download_button(
    label="⬇️ Download Rata-rata Cluster",
    data=csv_mean,
    file_name="ringkasan_cluster_mean.csv",
    mime="text/csv"
)

csv_median = cluster_median.to_csv().encode("utf-8")

st.download_button(
    label="⬇️ Download Median Cluster",
    data=csv_median,
    file_name="ringkasan_cluster_median.csv",
    mime="text/csv"
)


# ==========================================================
# 20. KESIMPULAN
# ==========================================================

st.header("17. Kesimpulan Praktikum")

st.markdown(
    f"""
    Berdasarkan proses clustering menggunakan algoritma
    **K-Means**, sebanyak **{len(hasil):,} data gempa bumi**
    berhasil dikelompokkan menjadi **{K_FINAL} cluster**.

    Hasil evaluasi model menunjukkan:

    - Jumlah data yang dianalisis: **{len(hasil):,}**
    - Jumlah fitur clustering: **{len(features)}**
    - Jumlah cluster akhir: **{K_FINAL}**
    - Inertia: **{final_inertia:.4f}**
    - Silhouette Score: **{final_silhouette:.4f}**

    Hasil pengelompokan dapat diamati melalui grafik,
    tabel rata-rata, median, serta jumlah anggota setiap cluster.

    Interpretasi cluster merupakan pengelompokan berdasarkan
    kemiripan data, bukan klasifikasi resmi tingkat bahaya gempa.
    """
)

st.success(
    "Seluruh proses analisis K-Means Clustering selesai."
)
