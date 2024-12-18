---

# Lung Cancer Prediction

This project aims to predict the likelihood of lung cancer using a dataset containing various health survey responses. The project involves exploratory data analysis (EDA), data preprocessing, feature engineering, model selection, and hyperparameter tuning to develop a classification model for predicting lung cancer.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Technologies Used](#technologies-used)
- [Steps Involved](#steps-involved)
  - [1. Data Import](#1-data-import)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Preprocessing](#3-preprocessing)
  - [4. Feature Selection and Engineering](#4-feature-selection-and-engineering)
  - [5. Modeling](#5-modeling)
  - [6. Hyperparameter Tuning](#6-hyperparameter-tuning)
- [Results](#results)
- [Conclusion](#conclusion)

## Project Overview

This project uses a dataset on lung cancer to predict the presence of cancer in individuals. The goal is to build a classification model that can accurately predict lung cancer based on survey data. The project leverages machine learning techniques including feature selection, model training, and hyperparameter tuning to achieve the best performance.

## Data

The dataset used in this project, `survey lung cancer.csv`, contains 16 columns with various health-related survey responses. The key target variable is `LUNG_CANCER`, which indicates the presence of lung cancer (`YES` or `NO`).

## Technologies Used

- **Python**: The programming language used for this project.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib/Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning tasks including model training, evaluation, and hyperparameter tuning.

## Steps Involved

### 1. Data Import
The dataset is imported using `pandas.read_csv()`, and basic information about the dataset is displayed.

```python
df = pd.read_csv('./survey lung cancer.csv')
df.head()
```

### 2. Exploratory Data Analysis (EDA)

#### Basic Data Inspection
We inspect the data types, check for missing values, and observe summary statistics.

```python
df.info()
df.describe()
```

#### Visualizing Lung Cancer Distribution by Gender
A bar chart visualizes the distribution of lung cancer cases by gender, showing a class imbalance.

```python
grouped_data = df.groupby(['GENDER', 'LUNG_CANCER']).size().unstack()
grouped_data.plot(kind='bar')
```

### 3. Preprocessing

#### Encoding Categorical Variables
Both `GENDER` and `LUNG_CANCER` are encoded using numeric labels.

```python
df['GENDER'] = df['GENDER'].map({'M':2, 'F': 1 })
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({"YES":2, "NO":1})
```

#### Scaling the Age Feature
The `AGE` feature is scaled using `StandardScaler` to improve model performance.

```python
scaler = StandardScaler()
df['age_scaled'] = scaler.fit_transform(df[['AGE']])
```

### 4. Feature Selection and Engineering

#### Correlation Matrix
A heatmap is used to visualize correlations between numerical features, helping with feature selection.

```python
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```

#### Recursive Feature Elimination (RFE)
RFE is used to select the top features for modeling.

```python
rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=7)
X_rfe = rfe.fit_transform(X_train, y_train)
```

### 5. Modeling

Several classification models are tested, including Support Vector Classifier (SVC), Decision Tree, and Random Forest, using 5-fold cross-validation to evaluate their performance.

```python
models = {
    'SVC': SVC(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier()
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
```

### 6. Hyperparameter Tuning

The `RandomForestClassifier` is tuned using `GridSearchCV` to find the best hyperparameters.

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

### Results

- The **Random Forest Classifier** performed the best among the models tested.
- Hyperparameter tuning improved the Random Forest model's performance.

### Conclusion

The project demonstrates the steps involved in building a machine learning model for predicting lung cancer using a survey dataset. By performing thorough data preprocessing, feature engineering, and hyperparameter tuning, we were able to develop an effective model. The results highlight the importance of handling class imbalance and tuning model parameters to achieve the best performance.

---
