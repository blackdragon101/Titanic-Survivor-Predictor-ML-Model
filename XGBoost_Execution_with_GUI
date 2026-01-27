!pip install xgboost gradio

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# --- Upload the dataset interactively ---
from google.colab import files
uploaded = files.upload()   # choose train.csv here

# Load train.csv
train_df = pd.read_csv("train.csv")

# ---------------------------
# Preprocessing
# ---------------------------
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Encode categorical variables
train_df = pd.get_dummies(train_df, columns=['Sex','Embarked'], drop_first=True)

# Features and target
X = train_df.drop(['Survived','PassengerId','Name','Ticket','Cabin'], axis=1)
y = train_df['Survived']

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Train XGBoost model
# ---------------------------
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
print("XGBoost Validation Accuracy:", accuracy_score(y_val, val_preds))

# ---------------------------
# Gradio UI
# ---------------------------
import gradio as gr

def predict_survival(Pclass, Age, SibSp, Parch, Fare, Sex, Embarked):
    # Build a single-row dataframe
    data = {
        'Pclass':[Pclass],
        'Age':[Age],
        'SibSp':[SibSp],
        'Parch':[Parch],
        'Fare':[Fare],
        'Sex_male':[1 if Sex=="male" else 0],
        'Embarked_Q':[1 if Embarked=="Q" else 0],
        'Embarked_S':[1 if Embarked=="S" else 0]
    }
    df_input = pd.DataFrame(data)
    prediction = model.predict(df_input)[0]
    return "✅ Survived" if prediction==1 else "❌ Did not survive"

demo = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Number(label="Pclass", info="Passenger class: 1 = First, 2 = Second, 3 = Third"),
        gr.Number(label="Age", info="Age of the passenger in years"),
        gr.Number(label="SibSp", info="Number of siblings/spouses aboard"),
        gr.Number(label="Parch", info="Number of parents/children aboard"),
        gr.Number(label="Fare", info="Ticket fare paid"),
        gr.Radio(["male","female"], label="Sex"),
        gr.Radio(["C","Q","S"], label="Embarked", info="Port of embarkation: C = Cherbourg, Q = Queenstown, S = Southampton")
    ],
    outputs="text",
    title="Titanic Survival Predictor (XGBoost)",
    description="Enter passenger details to predict survival. Pclass, SibSp, and Parch are explained below each field."
)

demo.launch()

