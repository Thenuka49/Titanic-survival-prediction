import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# -------------------- Custom Dark Theme Styling --------------------
st.markdown("""
<style>
/* Main App Background */
.main {
    background-color: #0e1117;
    color: #e0e0e0;
}

/* Sidebar Styling */
.sidebar .sidebar-content {
    background-color: #1c1f26;
    color: #e0e0e0;
}

/* Headers */
h1, h2, h3, h4 {
    color: #4cd4ff;
}

/* Dataframe Table Styling */
.dataframe {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
}

/* Text Inputs, Selects, Sliders */
.stTextInput>div>div>input, .stSelectbox>div>div>select, .stNumberInput>div>input, .stSlider {
    background-color: #2c2f38 !important;
    color: #ffffff !important;
}

/* Prediction Result */
.prediction-box {
    background-color: #1c1f26;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #4cd4ff;
    color: #ffffff;
}

/* Plotly Chart Background */
.js-plotly-plot .plotly {
    background-color: #0e1117 !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Load Dataset --------------------
@st.cache_data
def load_data():
    file_path = 'data/Titanic-Dataset.csv'
    if not os.path.exists(file_path):
        st.error(f"Dataset file not found at {file_path}")
        return None
    return pd.read_csv(file_path)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file 'model.pkl' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

df = load_data()
model = load_model()
if df is None or model is None:
    st.stop()

# -------------------- Preprocessing --------------------
def preprocess_data(df, for_prediction=False):
    X = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'] + (['Survived'] if not for_prediction else []),
                axis=1, errors='ignore')
    X['Age'] = X['Age'].fillna(X['Age'].median())
    X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    expected_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    return X[expected_cols]

# -------------------- App Title --------------------
st.title("🚢 Titanic Survival Prediction App")
st.write("This app predicts whether a passenger survived the Titanic disaster using a Random Forest model.")

# -------------------- Sidebar Navigation --------------------
st.sidebar.header("📌 Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Exploration", "Visualizations", "Prediction", "Model Performance"])

# -------------------- Data Exploration --------------------
if page == "Data Exploration":
    st.header("🔍 Data Exploration")
    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", list(df.columns))
    st.write("Data Types:\n", df.dtypes)

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Filter Data")
    pclass_filter = st.multiselect("Select Passenger Class", options=df['Pclass'].unique(),
                                   default=df['Pclass'].unique())
    filtered_df = df[df['Pclass'].isin(pclass_filter)]
    st.dataframe(filtered_df)

# -------------------- Visualizations --------------------
if page == "Visualizations":
    st.header("📊 Visualizations")

    st.subheader("Survival by Passenger Class")
    fig1 = px.histogram(df, x='Pclass', color='Survived', barmode='group', template="plotly_dark")
    st.plotly_chart(fig1)

    st.subheader("Survival by Sex")
    fig2 = px.histogram(df, x='Sex', color='Survived', barmode='group', template="plotly_dark")
    st.plotly_chart(fig2)

    st.subheader("Age Distribution")
    fig3 = px.histogram(df, x='Age', nbins=30, template="plotly_dark")
    st.plotly_chart(fig3)

# -------------------- Prediction --------------------
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }
    return preprocess_data(pd.DataFrame([data]), for_prediction=True)

if page == "Prediction":
    st.header("🎯 Make a Prediction")
    with st.form("prediction_form"):
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Sex", ['male', 'female'])
        age = st.slider("Age", 0, 100, 30)
        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
        embarked = st.selectbox("Embarked", ['C', 'Q', 'S'])
        submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                if age < 0 or fare < 0:
                    st.error("Age and Fare must be non-negative.")
                else:
                    input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0]
                    st.markdown(
                        f"<div class='prediction-box'>"
                        f"<b>Prediction:</b> {'🟢 Survived' if prediction == 1 else '🔴 Did Not Survive'}<br>"
                        f"<b>Survival Probability:</b> {probability[1]:.2%}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error: {e}")

# -------------------- Model Performance --------------------
@st.cache_data
def compute_performance_metrics():
    X = preprocess_data(df)
    y = df['Survived']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred),
        'cm': confusion_matrix(y_test, y_pred)
    }

if page == "Model Performance":
    st.header("📈 Model Performance")
    metrics = compute_performance_metrics()
    st.subheader("Metrics")
    st.write(f"**Accuracy:** {metrics['accuracy']:.4f}")
    st.text("Classification Report:")
    st.code(metrics['report'])

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
