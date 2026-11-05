import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

st.set_page_config(page_title="Rental Price & Category Predictor", page_icon="🏠", layout="wide")
st.title("🏠 Rental Price Prediction and Category Classification")
st.write("A supervised machine learning demo that predicts monthly rent and classifies the rental category.")

@st.cache_data
def load_data():
    return pd.read_csv("rental_data.csv")

df = load_data()
features = ["area_sqft", "bedrooms", "bathrooms", "location", "furnishing", "property_type"]
cat_cols = ["location", "furnishing", "property_type"]
num_cols = ["area_sqft", "bedrooms", "bathrooms"]
X = df[features]
y_rent = df["rent"]
y_cat = df["rent_category"]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("numeric", "passthrough", num_cols)
])
rent_model = Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(n_estimators=150, random_state=42))])
category_model = Pipeline([("prep", preprocessor), ("model", RandomForestClassifier(n_estimators=150, random_state=42))])

X_train, X_test, yr_train, yr_test = train_test_split(X, y_rent, test_size=0.2, random_state=42)
_, _, yc_train, yc_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
rent_model.fit(X_train, yr_train)
category_model.fit(X_train, yc_train)
rent_mae = mean_absolute_error(yr_test, rent_model.predict(X_test))
category_accuracy = accuracy_score(yc_test, category_model.predict(X_test))

st.sidebar.header("Enter property details")
area = st.sidebar.number_input("Area (sq. ft.)", min_value=100, max_value=10000, value=900, step=50)
bedrooms = st.sidebar.selectbox("Bedrooms", [1,2,3,4,5], index=1)
bathrooms = st.sidebar.selectbox("Bathrooms", [1,2,3,4,5], index=0)
location = st.sidebar.selectbox("Location", sorted(df.location.unique()))
furnishing = st.sidebar.selectbox("Furnishing", sorted(df.furnishing.unique()))
property_type = st.sidebar.selectbox("Property type", sorted(df.property_type.unique()))

if st.sidebar.button("Predict rental price"):
    sample = pd.DataFrame([[area, bedrooms, bathrooms, location, furnishing, property_type]], columns=features)
    predicted_rent = float(rent_model.predict(sample)[0])
    predicted_category = category_model.predict(sample)[0]
    c1,c2 = st.columns(2)
    c1.metric("Estimated monthly rent", f"₹{predicted_rent:,.0f}")
    c2.metric("Predicted rent category", predicted_category)
    st.info("This is a classroom demo trained on sample data. Actual rent depends on the city, neighbourhood, amenities, and current market conditions.")

st.divider()
st.subheader("Dataset preview")
st.dataframe(df.head(10), use_container_width=True)
st.subheader("Model evaluation")
c1,c2 = st.columns(2)
c1.metric("Rent model MAE", f"₹{rent_mae:,.0f}", help="Average absolute prediction error on the held-out sample test set.")
c2.metric("Category model accuracy", f"{category_accuracy:.1%}")
st.caption("Demo dataset contains generated sample records, not verified market listings. Replace rental_data.csv with a real, labelled dataset for meaningful results.")
