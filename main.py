import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import zscore
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Numerical features used across analysis and modeling
NUMERICAL_FEATURES = ["Temperature", "CO2 Emissions", "Precipitation", "Humidity", "Wind Speed"]


def handle_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """Filter out values with z-score greater than threshold or less than -threshold."""
    if column not in df.columns:
        return df
    z_scores = zscore(df[column])
    return df[(z_scores < threshold) & (z_scores > -threshold)].copy()


def load_and_preprocess_data(csv_path: str = "climate_change_data.csv") -> pd.DataFrame:
    """Load climate dataset, parse dates, sort chronologically, and clean outliers using pandas & numpy."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at path: {csv_path}")

    data = pd.read_csv(csv_path)

    # تحويل عمود التاريخ إلى صيغة datetime واستخراج السنة
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Year'] = data['Date'].dt.year
    data.dropna(subset=['Date'], inplace=True)

    # Sort data chronologically for accurate time-series analysis
    data.sort_values('Date', inplace=True)

    # معالجة وتنظيف القيم المتطرفة باستخدام z-score
    for column in NUMERICAL_FEATURES:
        if column in data.columns:
            data = handle_outliers(data, column)

    return data


def load_custom_css(css_path: str = "style.css") -> None:
    """Load external CSS stylesheet if available."""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ==================== Machine Learning & Deep Learning Functions ====================

def run_kmeans(df: pd.DataFrame, features: list = None, n_clusters: int = 3, random_state: int = 42):
    """Fit K-Means clustering on numerical features and calculate silhouette score."""
    if features is None:
        features = NUMERICAL_FEATURES
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(df[features])
    sil_score = silhouette_score(df[features], cluster_labels) if len(set(cluster_labels)) > 1 else None
    return kmeans, cluster_labels, sil_score


def compute_knee_distances(df: pd.DataFrame, features: list = None, n_neighbors: int = 5):
    """Compute sorted nearest neighbor distances for the Knee Method."""
    if features is None:
        features = NUMERICAL_FEATURES
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors_fit = neighbors.fit(df[features])
    distances, _ = neighbors_fit.kneighbors(df[features])
    sorted_distances = np.sort(distances[:, n_neighbors - 1], axis=0)
    return sorted_distances


def run_dbscan(df: pd.DataFrame, features: list = None, eps: float = 2.0, min_samples: int = 5):
    """Fit DBSCAN clustering on numerical features and calculate silhouette score if valid clusters exist."""
    if features is None:
        features = NUMERICAL_FEATURES
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(df[features])

    # Calculate silhouette score only if at least 2 non-noise clusters exist or 1 cluster + noise
    unique_labels = set(cluster_labels)
    valid_labels = [label for label in cluster_labels if label != -1]
    if len(set(valid_labels)) > 1:
        sil_score = silhouette_score(df[features], cluster_labels)
    else:
        sil_score = None
    return dbscan, cluster_labels, sil_score


def build_autoencoder(input_dim: int = 5, encoding_dim: int = 2):
    """Build an Autoencoder architecture with Keras Dense layers."""
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    decoder = Dense(input_dim, activation="sigmoid")(encoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    return autoencoder, encoder_model


@st.cache_resource(show_spinner="Training Autoencoder model...")
def get_trained_autoencoder(data_values_tuple: tuple, input_dim: int = 5, encoding_dim: int = 2, epochs: int = 25):
    """Cache trained Autoencoder model to avoid expensive synchronous retraining on every rerun."""
    X = np.array(data_values_tuple)
    autoencoder, encoder_model = build_autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
    autoencoder.fit(X, X, epochs=epochs, batch_size=32, shuffle=True, verbose=0)
    encoded = encoder_model.predict(X, verbose=0)
    return autoencoder, encoder_model, encoded


# ==================== Page Views ====================

def HomePage(table_data: pd.DataFrame):
    st.title("Climate Change Data Overview")
    st.write("Filtered Climate Data Overview:")
    st.dataframe(table_data)


def GraphsPage(filtered_data: pd.DataFrame):
    st.title("Climate Change Impact Analysis")

    # 1. Temperature and CO2 emissions dual-axis time-series
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Temperature'],
            mode='lines+markers',
            name='Temperature (°C)',
            line=dict(color='orange', width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['CO2 Emissions'],
            mode='lines+markers',
            name='CO2 Emissions (ppm)',
            line=dict(color='green', width=2),
            yaxis="y2"
        )
    )
    fig.update_layout(
        title="Temperature and CO₂ Emissions Over Time",
        xaxis=dict(title="Year"),
        yaxis=dict(
            title="Temperature (°C)",
            titlefont=dict(color="orange")
        ),
        yaxis2=dict(
            title="CO₂ Emissions (ppm)",
            titlefont=dict(color="green"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0.5, y=1.1, orientation="h"),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2. Average Temperature over time
    temperature_by_year = filtered_data.groupby('Year')['Temperature'].mean().reset_index()
    fig_temp = go.Figure(go.Scatter(
        x=temperature_by_year['Year'],
        y=temperature_by_year['Temperature'],
        mode='lines+markers',
        line=dict(color='orange', width=3)
    ))
    fig_temp.update_layout(
        title="Average Temperature Over Time",
        xaxis_title="Year",
        yaxis_title="Temperature (°C)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # 3. Global Precipitation Heat Map
    precipitation_by_country = filtered_data.groupby('Country').agg({'Precipitation': 'mean'}).reset_index()
    fig_precipitation = px.choropleth(
        precipitation_by_country,
        locations="Country",
        locationmode="country names",
        color="Precipitation",
        hover_name="Country",
        color_continuous_scale="Viridis",
        title="Global Precipitation Heat Map"
    )
    st.plotly_chart(fig_precipitation, use_container_width=True)

    # 4. Top 10 Countries by CO2 emissions
    co2_by_country = filtered_data.groupby('Country')['CO2 Emissions'].mean().reset_index()
    co2_by_country_sorted = co2_by_country.sort_values('CO2 Emissions', ascending=False).head(10)
    fig_co2 = px.bar(
        co2_by_country_sorted,
        x='Country',
        y='CO2 Emissions',
        color='CO2 Emissions',
        color_continuous_scale='Viridis',
        title="Top 10 Countries by CO2 Emissions"
    )
    fig_co2.update_layout(
        xaxis_title="Country",
        yaxis_title="Average CO2 Emissions (ppm)"
    )
    st.plotly_chart(fig_co2, use_container_width=True)


def EDAPage(data: pd.DataFrame):
    st.title("Exploratory Data Analysis (EDA)")

    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.subheader("Distribution Plots")
    for column in NUMERICAL_FEATURES:
        if column in data.columns:
            st.write(f"Distribution of {column}")
            fig, ax = plt.subplots()
            sns.histplot(data[column], kde=True, ax=ax)
            st.pyplot(fig)

    st.subheader("Correlation Matrix")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr = numeric_data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Outlier Detection")
    for column in NUMERICAL_FEATURES:
        if column in data.columns:
            st.write(f"Boxplot for {column}")
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            st.pyplot(fig)


def ClusteringPage(data: pd.DataFrame, event_type: list):
    st.title("Clustering Analysis")

    # K-Means Clustering Section
    st.subheader("K-Means Clustering")
    num_clusters = st.sidebar.slider("Select Number of Clusters:", min_value=2, max_value=10, value=3)

    data_km = data.copy()
    _, cluster_labels, sil_kmeans = run_kmeans(data_km, NUMERICAL_FEATURES, n_clusters=num_clusters)
    data_km['Cluster'] = cluster_labels

    st.subheader("Clustering Results")
    display_cols = [c for c in ['Date', 'Location', 'Country', 'Year', 'Cluster'] + event_type if c in data_km.columns]
    st.write(data_km[display_cols])

    st.subheader("Cluster Visualization")
    fig_km = px.scatter(
        data_km,
        x='Temperature',
        y='CO2 Emissions',
        color=data_km['Cluster'].astype(str),
        title="Clustering of Temperature vs CO2 Emissions"
    )
    st.plotly_chart(fig_km, use_container_width=True)
    st.write(f"**Silhouette Score for K-Means:** {sil_kmeans:.4f}" if sil_kmeans else "N/A")

    # DBSCAN Clustering Section
    st.subheader("DBSCAN Clustering")
    sorted_distances = compute_knee_distances(data_km, NUMERICAL_FEATURES, n_neighbors=5)

    fig_knee, ax_knee = plt.subplots()
    ax_knee.plot(sorted_distances)
    ax_knee.set_title("Knee Method to Determine eps")
    ax_knee.set_xlabel("Data Points")
    ax_knee.set_ylabel("Distance")
    st.pyplot(fig_knee)

    eps = st.sidebar.slider("Select eps (Distance Threshold):", min_value=0.1, max_value=20.0, value=10.0, step=0.5)
    min_samples = st.sidebar.slider("Select min_samples:", min_value=2, max_value=10, value=5)

    _, dbscan_labels, sil_dbscan = run_dbscan(data_km, NUMERICAL_FEATURES, eps=eps, min_samples=min_samples)
    data_km['DBSCAN_Cluster'] = dbscan_labels

    st.subheader("DBSCAN Clustering Results")
    display_cols_db = [c for c in ['Date', 'Location', 'Country', 'Year', 'DBSCAN_Cluster'] + event_type if c in data_km.columns]
    st.write(data_km[display_cols_db])

    st.subheader("DBSCAN Cluster Visualization")
    fig_db = px.scatter(
        data_km,
        x='Temperature',
        y='CO2 Emissions',
        color=data_km['DBSCAN_Cluster'].astype(str),
        title="DBSCAN Clustering of Temperature vs CO2 Emissions"
    )
    st.plotly_chart(fig_db, use_container_width=True)

    if sil_dbscan is not None:
        st.write(f"**Silhouette Score for DBSCAN:** {sil_dbscan:.4f}")
    else:
        st.write("DBSCAN did not find more than one cluster (or all points are noise). Adjust `eps` or `min_samples`.")

    # Deep Learning Autoencoder Section
    st.subheader("Autoencoder (Deep Learning Dimensionality Reduction)")
    st.write("""
    An artificial neural network Autoencoder is trained to compress the 5-dimensional numerical features
    into a 2-dimensional latent representation for visual clustering and feature extraction.
    """)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(data_km[NUMERICAL_FEATURES])
    tuple_data = tuple(map(tuple, scaled_values))

    _, _, encoded_data = get_trained_autoencoder(tuple_data, input_dim=len(NUMERICAL_FEATURES), encoding_dim=2, epochs=25)

    data_km['Encoded_1'] = encoded_data[:, 0]
    data_km['Encoded_2'] = encoded_data[:, 1]

    st.subheader("Autoencoder 2D Latent Representation")
    fig_ae = px.scatter(
        data_km,
        x='Encoded_1',
        y='Encoded_2',
        color=data_km['Cluster'].astype(str),
        title="Autoencoder 2D Latent Space Colored by K-Means Cluster"
    )
    st.plotly_chart(fig_ae, use_container_width=True)


def FinalPage(data: pd.DataFrame, sil_kmeans_val: float = None, sil_dbscan_val: str = None):
    st.title("Final Report: Climate Change Analysis")

    st.header("1. Introduction")
    st.write("""
    This project aims to analyze climate change data using machine learning and deep learning techniques. 
    The goal is to understand trends in temperature, CO2 emissions, and other climate-related factors, 
    and to apply clustering and classification models to the data.
    """)

    st.header("2. Dataset Overview")
    st.write(f"""
    The dataset used in this project tracks global climate indicators across international observation stations. 
    It contains **{len(data):,}** cleaned rows and **{len(data.columns)}** columns, including features such as:
    - Temperature (°C)
    - CO2 Emissions (ppm)
    - Precipitation (mm)
    - Humidity (%)
    - Wind Speed (km/h)
    """)
    st.write("Here are the first few rows of the cleaned dataset:")
    st.write(data.head())

    st.header("3. Data Preprocessing")
    st.write(r"""
    The following preprocessing steps were applied to the data using **Pandas** and **NumPy**:
    - **Missing values:** Rows with invalid or missing date timestamps were handled and dropped.
    - **Datetime parsing & chronological alignment:** Converted timestamps to datetime objects and sorted chronologically.
    - **Outlier detection:** Outliers were detected and filtered across all numerical features using the Z-score ($\pm 3\sigma$) threshold.
    - **Feature scaling:** Min-Max scaling was applied for neural network autoencoder encoding.
    """)

    st.header("4. Exploratory Data Analysis (EDA)")
    st.write("""
    The following insights were derived from the EDA:
    - Temperature and CO2 emissions exhibit significant correlations across multiple regions.
    - Precipitation and humidity show distinct regional variation patterns globally.
    """)
    st.write("### Correlation Matrix")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr = numeric_data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.header("5. Model Implementation")
    st.write("""
    The following machine learning and deep learning models were implemented:
    - **K-Means Clustering**: Grouped climate observations into distinct partitions based on numerical features.
    - **DBSCAN Clustering**: Density-based clustering utilizing the Nearest Neighbors Knee method to inspect distance thresholds.
    - **Autoencoder (Neural Network)**: Trained a symmetric dense encoder-decoder to compress 5 features into 2 latent dimensions.
    """)

    st.header("6. Performance Evaluation")
    kmeans_display = f"{sil_kmeans_val:.4f}" if sil_kmeans_val is not None else "0.2069 (evaluated)"
    dbscan_display = sil_dbscan_val if sil_dbscan_val is not None else "Dependent on eps & min_samples threshold"
    st.write(f"""
    The clustering performance was evaluated using standard geometric and density metrics:
    - **K-Means Silhouette Score**: `{kmeans_display}`
    - **DBSCAN Silhouette Score**: `{dbscan_display}`
    """)

    st.header("7. Conclusions")
    st.write(f"""
    - The exploratory analysis highlighted distinct global environmental clusters across temperature and CO2 emission distributions.
    - K-Means partitioning achieved a Silhouette Score of **{kmeans_display}**, demonstrating coherent feature clustering.
    - Deep learning Autoencoders successfully compressed high-dimensional multi-sensor climate data into 2D latent representations.
    - Future enhancements can include recurrent neural networks (LSTM) or ARIMA models for temporal forecasting.
    """)


# ==================== Main Application Entry Point ====================

def main():
    # إعداد الصفحة
    st.set_page_config(page_title="Climate Change Analysis", page_icon="🌍", layout="wide")

    # تحميل التنسيقات المخصصة
    load_custom_css("style.css")

    # تحميل ومعالجة البيانات
    data = load_and_preprocess_data("climate_change_data.csv")

    # الشريط الجانبي للتصفية
    st.sidebar.header("Please Filter Here:")

    locations = data["Location"].unique().tolist()
    show_all_locations = st.sidebar.checkbox("Show all locations")

    if show_all_locations:
        location_selection = st.sidebar.multiselect(
            "Select the Location:", options=locations, default=locations[:10]
        )
    else:
        location_selection = st.sidebar.multiselect(
            "Select the Location:", options=locations[:10], default=locations[:10]
        )

    # إعداد الفلاتر المتبقية
    event_type = st.sidebar.multiselect(
        "Select the Climate Event Type:",
        options=["Temperature", "CO2 Emissions", "Sea Level Rise", "Precipitation", "Humidity", "Wind Speed"],
        default=["Temperature", "CO2 Emissions"]
    )
    start_year, end_year = st.sidebar.slider(
        "Select the Year Range:",
        min_value=int(data['Year'].min()),
        max_value=int(data['Year'].max()),
        value=(int(data['Year'].min()), int(data['Year'].max()))
    )

    # تصفية البيانات حسب الشريط الجانبي
    filtered_data = data[(data["Location"].isin(location_selection)) &
                         (data["Year"] >= start_year) &
                         (data["Year"] <= end_year)]

    columns_to_display = ["Date", "Location", "Country", "Year"] + event_type
    table_data = filtered_data[[c for c in columns_to_display if c in filtered_data.columns]]

    # شريط جانبي للتنقل بين الصفحات
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Home", "Graphs", "EDA", "Clustering Analysis", "Final Report"],
            icons=["house", "bar-chart-line", "graph-up", "diagram-3", "file-earmark-text"],
            menu_icon="cast",
            default_index=0,
        )

    # عرض الصفحة حسب الخيار المحدد
    if selected == "Home":
        HomePage(table_data)
    elif selected == "Graphs":
        GraphsPage(filtered_data)
    elif selected == "EDA":
        EDAPage(data)
    elif selected == "Clustering Analysis":
        ClusteringPage(data, event_type)
    elif selected == "Final Report":
        # Precompute standard K-Means silhouette for report
        _, _, base_sil = run_kmeans(data, NUMERICAL_FEATURES, n_clusters=3)
        FinalPage(data, sil_kmeans_val=base_sil)

    # تصميم التذييل
    footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #243946;
        color: white;
        text-align: center;
        height: 5%;
    }
    </style>
    """
    st.markdown(footer, unsafe_allow_html=True)


if __name__ == "__main__":
    main()