import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Load and prepare data
df = pd.read_excel('Bankruptcy.xlsx')
df['class'] = df['class'].map({'non-bankruptcy': 0, 'bankruptcy': 1})

X = df.drop(columns=['class'])
y = df['class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=GaussianNB()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

st.title("Bankruptcy Prediction - Naive Bayes")

# Create tabs
tab1, tab2 = st.tabs(["EDA & Visualization", "Predict Bankruptcy"])

with tab1:
    st.header("EDA and Model Evaluation")

    df = pd.read_excel('Bankruptcy.xlsx')

    df.hist(figsize=(10,12))
    plt.suptitle('Distribution of risk levels')
    st.pyplot(plt.gcf())
    plt.clf()

    df['class'].value_counts().plot(kind='bar')
    plt.title('Number of Companies by Class')
    plt.xlabel('Class')
    plt.ylabel('Count')
    st.pyplot(plt)
    plt.clf()

    df['class'] = df['class'].map({'non-bankruptcy': 0, 'bankruptcy': 1})

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)
    plt.clf()

    plt.figure(figsize=(10, 6))
    df.boxplot(column=['industrial_risk', 'management_risk', 'financial_flexibility',
                    'credibility', 'competitiveness', 'operating_risk'],color='blue')
    plt.title("Boxplots of Risk Factors")
    plt.ylabel("Risk Level")
    st.pyplot(plt)
    plt.clf()

    pair_fig=sns.pairplot(df, hue='class')
    st.pyplot(pair_fig)
    plt.clf()

    X=df.drop(columns=['class'])
    y=df['class']

    X_new = SelectKBest(score_func=f_classif, k='all').fit(X, y)
    anova_scores = X_new.scores_
    mi = SelectKBest(mutual_info_classif, k='all').fit(X, y)
    mi_scores = mi.scores_

    scores_df = pd.DataFrame({'Feature': X.columns,
                            'ANOVA Score': anova_scores,
                            'Mutual Info': mi_scores})
    st.write(scores_df.sort_values(by='ANOVA Score', ascending=False))

    st.subheader("📊 Model Evaluation Results")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.text("Classification Report:\n" + classification_report(y_test, y_pred))
    st.write("**Confusion Matrix:**")
    st.write(confusion_matrix(y_test, y_pred))

    y_pred1 = model.predict(X_test)
    y_pred1_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred1_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line for random guessing
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)
    plt.clf()

with tab2:
    st.header("Predict Bankruptcy from User Inputs")

    # Create input fields dynamically based on feature columns
    input_data = {}
    for feature in X.columns:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        input_data[feature] = st.selectbox(f"{feature}", options= [0, 0.5, 1], index=0)

    # Convert input to dataframe
    input_df = pd.DataFrame([input_data])

    # Predict button
    if st.button("Predict"):
        result = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of bankruptcy

        if result == 1:
            st.error(f"The company is **Bankrupt** (Probability: {probability:.2f})")
        else:
            st.success(f"The company is **Non-Bankrupt** (Probability: {1-probability:.2f})")
