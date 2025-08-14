# Titanic Survival Prediction App

This is a Streamlit web application that predicts whether a passenger survived the Titanic disaster using a Random Forest model. The app also allows dataset exploration, visualization, and checking model performance.

---

## Features

- Explore dataset: view shape, columns, types, and sample records  
- Visualize survival by passenger class, sex, and age  
- Predict survival for custom passenger data  
- Check model performance: accuracy, classification report, confusion matrix  
- Custom dark theme interface  

---

## Installation & Usage

1. Clone the repository:

git clone https://github.com/your-username/titanic-survival-app.git
cd titanic-survival-app
(Optional) Create virtual environment:

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
Install dependencies:

pip install -r requirements.txt
Ensure required files exist:

data/Titanic-Dataset.csv

model.pkl

Run the app:

streamlit run app.py
Dataset
File: data/Titanic-Dataset.csv

Features: Pclass, Name, Sex, Age, SibSp, Parch, Fare, Embarked, Survived

Target: Survived (0 = Did not survive, 1 = Survived)

Model
Algorithm: Random Forest Classifier

File: model.pkl

Input features: Pclass, Age, SibSp, Parch, Fare, FamilySize, Sex_male, Embarked_Q, Embarked_S

Preprocessing
Drop columns: PassengerId, Name, Ticket, Cabin

Fill missing Age with median

One-hot encode Sex and Embarked

Add FamilySize = SibSp + Parch + 1

Ensure all required input columns exist

Pages
Data Exploration – inspect dataset, filter by passenger class

Visualizations – interactive plots of survival

Prediction – predict survival for custom input

Model Performance – accuracy, report, confusion matrix

Dependencies
Python 3.8+

streamlit, pandas, numpy, scikit-learn, plotly, seaborn, matplotlib

Install dependencies:


pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib