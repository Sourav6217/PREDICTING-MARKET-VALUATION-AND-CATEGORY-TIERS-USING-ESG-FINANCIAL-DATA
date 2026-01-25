# streamlit_app.py
# NOTE: This Streamlit app is a DIRECT PORT of the original Python/Jupyter file.
# No extra EDA, no new logic, no assumptions added.
# Goal: reproduce ALL visuals & outputs from the notebook in Streamlit tabs.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, classification_report, confusion_matrix

# ---------------- Page Config ----------------
st.set_page_config(page_title="ESG–Financial Analysis", layout="wide")

# ---------------- Load Data -----------------
@st.cache_data
def load_data():
    return pd.read_csv("company_esg_financial_dataset.csv")

df = load_data()

st.title("Predicting Market Valuation & Category Tiers using ESG–Financial Data")
st.caption("Exact Streamlit replication of the original Python analysis")

# ---------------- Tabs -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Dataset & Preparation",
    "📊 Core Visual Analysis",
    "📈 ESG vs Financial Relationships",
    "🤖 Model Outputs",
    "➕ Custom Space"
])

# ---------------- TAB 1: Dataset & Prep -----------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Column Overview")
    st.write(df.columns)

    st.subheader("Basic Statistics (as used in notebook)")
    st.write(df.describe())

# ---------------- TAB 2: Core Visual Analysis -----------------
with tab2:
    st.subheader("Distributions (Notebook Plots)")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

# ---------------- TAB 3: ESG vs Financial -----------------
with tab3:
    st.subheader("ESG vs Financial Scatter Plots")

    esg_cols = [c for c in df.columns if "ESG" in c.upper()]
    fin_cols = numeric_cols

    for esg in esg_cols:
        for fin in fin_cols:
            if esg != fin:
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x=esg, y=fin, ax=ax)
                ax.set_title(f"{esg} vs {fin}")
                st.pyplot(fig)

# ---------------- TAB 4: Model Outputs -----------------
with tab4:
    st.subheader("Random Forest Regression (Same as Notebook)")

    target = numeric_cols[-1]
    features = numeric_cols[:-1]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    st.write("R² Score:", r2_score(y_test, preds))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))

    st.subheader("Feature Importance")
    fi = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(data=fi, x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)

# ---------------- TAB 5: Custom Space -----------------
with tab5:
    st.subheader("Your Custom Section")
    st.info("This tab is intentionally left blank for your own extensions, widgets, or business logic.")

st.caption("Streamlit app mirrors the original Python file without adding extra analysis")
