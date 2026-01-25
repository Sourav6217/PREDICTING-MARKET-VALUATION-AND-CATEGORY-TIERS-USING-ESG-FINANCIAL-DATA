# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="ESG & Financial Analytics Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("company_esg_financial_dataset.csv")

df = load_data()

st.title("📊 ESG & Financial Performance – Interactive Dashboard")

# ---------------- Tabs -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Data Overview",
    "📈 Exploratory Data Analysis",
    "🌱 ESG vs Financials",
    "🤖 ML Models",
    "🎛 Interactive Playground"
])

# -------- Tab 1: Data Overview ---------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# -------- Tab 2: EDA ---------
with tab2:
    st.subheader("Distribution Plots")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    col = st.selectbox("Select column", numeric_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------- Tab 3: ESG vs Financial ---------
with tab3:
    st.subheader("ESG vs Financial Scatter")

    esg_col = st.selectbox("Select ESG Metric", [c for c in df.columns if "ESG" in c.upper()])
    fin_col = st.selectbox("Select Financial Metric", numeric_cols)

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=esg_col, y=fin_col, ax=ax)
    st.pyplot(fig)

# -------- Tab 4: ML Models ---------
with tab4:
    st.subheader("Random Forest Regression")

    target = st.selectbox("Target Variable", numeric_cols)
    features = st.multiselect("Feature Variables", [c for c in numeric_cols if c != target])

    if len(features) > 0:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        st.metric("R² Score", round(r2, 3))
        st.metric("RMSE", round(rmse, 3))

        st.subheader("Feature Importance")
        fi = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
        fi = fi.sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(data=fi, x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)

# -------- Tab 5: Interactive Widget ---------
with tab5:
    st.subheader("What-if Analysis Widget")
    st.write("Adjust ESG score to see expected impact on a financial metric")

    slider_val = st.slider("ESG Score", float(df[esg_col].min()), float(df[esg_col].max()))
    st.write("Selected ESG Score:", slider_val)

    st.info("This widget can be extended with trained models for live prediction.")

st.caption("Built from Jupyter Notebook → Streamlit | End-to-End Interactive App")

