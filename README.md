# Titanic Survival Predictor 🚢⚡

An interactive machine learning project that predicts whether a passenger on the Titanic would have survived, using the **XGBoost algorithm** and a **Gradio UI** for easy input. For comparison code using Random Forest ML Algorithm is also provided.

---

## Overview
This project demonstrates:
- **Data preprocessing**: Handling missing values, encoding categorical variables, and preparing features.
- **Model training**: Using XGBoost, a powerful gradient boosting algorithm, to achieve high accuracy.
- **Interactive UI**: Built with Gradio, allowing users to enter passenger details and instantly see survival predictions.

---

## Features
- **Upload Titanic dataset (`train.csv`)** directly in Google Colab.
- **Preprocessing pipeline**: Fill missing values, encode categorical features, drop irrelevant columns.
- **XGBoost model**: Trained on passenger data to predict survival.
- **Gradio interface**: User‑friendly form with explanations for inputs:
  - **Pclass**: Passenger class (1 = First, 2 = Second, 3 = Third)  
  - **SibSp**: Number of siblings/spouses aboard  
  - **Parch**: Number of parents/children aboard  
  - **Fare**: Ticket fare paid  
  - **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)  

---

## How to Run
1. Open the notebook in **Google Colab**.
2. Install dependencies:
   ```bash
   !pip install xgboost gradio
   ```
3. Upload `train.csv` when prompted.
4. Run all cells to train the model.
5. Launch the Gradio UI — you’ll get a link to interact with the predictor.

---

## Example Usage
Enter passenger details in the Gradio form:
- Pclass = 3  
- Age = 22  
- SibSp = 1  
- Parch = 0  
- Fare = 7.25  
- Sex = male  
- Embarked = S  

 Output: **❌ Did not survive**

---
Screenshots:
<img width="959" height="415" alt="Screenshot 2026-01-27 233654" src="https://github.com/user-attachments/assets/baee0a90-a787-4da9-a7b5-c4ffadf525eb" />

<img width="949" height="386" alt="Screenshot 2026-01-27 233724" src="https://github.com/user-attachments/assets/120885f3-503e-4540-8ffb-0fa4efbc617c" />


---

## Accuracy
The model achieves strong validation accuracy on the Titanic dataset, thanks to XGBoost’s sequential learning approach.

---

## Future Improvements 
- Compare performance with other algorithms (like Logistic Regression).  
- Enhance UI with visualizations of survival probabilities.

---

## Contribution
Pull requests are welcome! If you’d like to add new features or improve the UI, feel free to fork the repo and submit changes.

---
