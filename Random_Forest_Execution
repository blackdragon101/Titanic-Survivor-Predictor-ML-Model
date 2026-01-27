# Upload both train.csv and test.csv
from google.colab import files
uploaded = files.upload()   # choose train.csv and test.csv

import pandas as pd

# Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# ---------------------------
# Preprocessing function
# ---------------------------
def preprocess(df, is_train=True):
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Encode categorical
    df = pd.get_dummies(df, columns=['Sex','Embarked'], drop_first=True)

    # Scale numerical
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])

    return df

# Preprocess train and test
train_df = preprocess(train_df)
test_df = preprocess(test_df, is_train=False)

# ---------------------------
# Train/Test split
# ---------------------------
X = train_df.drop(['Survived','PassengerId','Name','Ticket','Cabin'], axis=1)
y = train_df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Train a ML model
# ---------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))

# ---------------------------
# Apply model to test.csv
# ---------------------------
X_test = test_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test_preds = model.predict(X_test)

# Save predictions
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': test_preds})
output.to_csv("ml_predictions.csv", index=False)
print("ML predictions saved to ml_predictions.csv")

# Download predictions
files.download("ml_predictions.csv")

