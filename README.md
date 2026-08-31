# ❤️ Heart Disease Analysis & Data Preprocessing

## 📌 Project Overview

This project performs **Exploratory Data Analysis (EDA)** and **data preprocessing** on a heart disease dataset containing **918 records and 12 features**.

The analysis focuses on understanding the structure and distribution of the dataset, checking data quality, exploring the target variable, examining relationships between numerical features, and preparing categorical and numerical variables for potential machine learning applications.

The project is implemented in **Python using Jupyter Notebook**.

---

## 🎯 Objectives

The main objectives of this project are:

* Understand the structure of the heart disease dataset.
* Perform exploratory data analysis (EDA).
* Check for missing values and duplicate records.
* Analyze the distribution of the target variable.
* Examine correlations between numerical features.
* Convert categorical variables into numerical representations.
* Scale numerical features using standardization.
* Prepare a clean dataset suitable for further machine learning work.

---

## 📊 Dataset

The dataset is loaded from:

```text
heart.csv
```

It contains:

* **918 rows**
* **12 original columns**
* **6 integer columns**
* **1 floating-point column**
* **5 categorical/object columns**

### Features

| Feature          | Description                           |
| ---------------- | ------------------------------------- |
| `Age`            | Age of the individual                 |
| `Sex`            | Sex of the individual                 |
| `ChestPainType`  | Type of chest pain                    |
| `RestingBP`      | Resting blood pressure                |
| `Cholesterol`    | Cholesterol level                     |
| `FastingBS`      | Fasting blood sugar indicator         |
| `RestingECG`     | Resting electrocardiographic results  |
| `MaxHR`          | Maximum heart rate achieved           |
| `ExerciseAngina` | Exercise-induced angina indicator     |
| `Oldpeak`        | ST depression value                   |
| `ST_Slope`       | Slope of the peak exercise ST segment |
| `HeartDisease`   | Target variable                       |

The notebook confirms that all 918 records are non-null across the 12 columns.

---

## 🔍 Exploratory Data Analysis

### 1. Dataset Structure

The dataset contains:

```text
918 rows × 12 columns
```

The numerical variables include:

* Age
* RestingBP
* Cholesterol
* FastingBS
* MaxHR
* Oldpeak
* HeartDisease

The remaining variables are categorical.

---

### 2. Statistical Summary

Descriptive statistics were calculated using `df.describe()`.

Some key statistics include:

| Feature     |   Mean | Minimum | Maximum |
| ----------- | -----: | ------: | ------: |
| Age         |  53.51 |      28 |      77 |
| RestingBP   | 132.40 |       0 |     200 |
| Cholesterol | 198.80 |       0 |     603 |
| FastingBS   |   0.23 |       0 |       1 |
| MaxHR       | 136.81 |      60 |     202 |
| Oldpeak     |   0.89 |    -2.6 |     6.2 |

The target variable `HeartDisease` has a mean of approximately **0.553**, reflecting the proportion of records labeled `1`.

---

### 3. Missing Value Analysis

Missing values were checked using:

```python
df.isnull().sum()
```

The analysis shows **0 missing values in every column**.

Therefore, no missing-value imputation was required.

---

### 4. Duplicate Analysis

Duplicate records were checked using:

```python
df.duplicated().sum()
```

Result:

```text
0
```

This indicates that no duplicate rows were identified in the dataset.

---

### 5. Target Variable Distribution

The target variable is:

```text
HeartDisease
```

Its distribution is:

| Heart Disease | Count |
| ------------- | ----: |
| `0`           |   410 |
| `1`           |   508 |

The notebook also visualizes this distribution using a bar chart.

---

### 6. Correlation Analysis

A correlation matrix was generated for numerical variables using:

```python
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)
```

This heatmap helps identify relationships between numerical variables and the target variable.

---

# 🧹 Data Preprocessing

## 1. One-Hot Encoding

Categorical variables were converted into numerical variables using:

```python
df_encode = pd.get_dummies(df, drop_first=True)
```

Using `drop_first=True` removes one category from each categorical feature to avoid redundant dummy variables.

After encoding, the dataset contains:

```text
918 rows × 16 columns
```

New encoded features include:

* `Sex_M`
* `ChestPainType_ATA`
* `ChestPainType_NAP`
* `ChestPainType_TA`
* `RestingECG_Normal`
* `RestingECG_ST`
* `ExerciseAngina_Y`
* `ST_Slope_Flat`
* `ST_Slope_Up`

---

## 2. Converting Boolean Values

The encoded Boolean values were converted into integers:

```python
df_encode = df_encode.astype(int)
```

This produces numerical `0` and `1` representations suitable for subsequent machine learning workflows.

---

## 3. Feature Scaling

The following numerical features were standardized:

```python
numerical_cols = [
    'Age',
    'RestingBP',
    'Cholesterol',
    'MaxHR',
    'Oldpeak'
]
```

The project uses `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_encode[numerical_cols] = scaler.fit_transform(
    df_encode[numerical_cols]
)
```

Standardization transforms the numerical features to a common scale, which can be useful for machine learning algorithms that are sensitive to feature magnitude.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Feature preprocessing and standardization
* **Jupyter Notebook** – Development environment

The notebook imports NumPy, Matplotlib, Seaborn, Pandas, and warnings handling at the beginning of the project.

---

# 🔄 Project Workflow

```text
        ┌───────────────────┐
        │   Load Dataset    │
        │     heart.csv     │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   Data Inspection  │
        │ Shape / Info /     │
        │ Statistics         │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │       EDA         │
        │ Target Analysis   │
        │ Correlation       │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │  Data Quality     │
        │ Missing Values    │
        │ Duplicate Check   │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Categorical       │
        │ Encoding          │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Feature Scaling   │
        │ StandardScaler    │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Preprocessed Data │
        └───────────────────┘
```

---

# 📈 Key Findings

Based on the notebook:

* The dataset contains **918 observations and 12 original features**.
* There are **no missing values**.
* There are **no duplicate records**.
* The target variable contains **508 positive (`1`) and 410 negative (`0`) observations**.
* Categorical features were successfully transformed using one-hot encoding.
* The encoded dataset contains **16 columns**.
* Numerical features were standardized using `StandardScaler`.
* A correlation heatmap was created to examine relationships among numerical variables.

## These findings are directly derived from the notebook's executed outputs.

# 📁 Project Structure

```text
Heart-Disease-Analysis/
│
├── heart.csv
├── Project2.ipynb
└── README.md
```

---

# 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd Heart-Disease-Analysis
```

### 3. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
Project2.ipynb
```

Make sure `heart.csv` is available in the appropriate project directory before running the notebook.

---

# 🔮 Future Improvements

The current notebook focuses on **EDA and preprocessing**. Possible extensions include:

* Train classification models such as Logistic Regression, Decision Tree, Random Forest, and SVM.
* Split the data into training and testing sets.
* Evaluate models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
* Generate a confusion matrix.
* Perform feature selection.
* Tune model hyperparameters.
* Compare multiple classification algorithms.
* Deploy the best-performing model as a simple prediction application.

> **Note:** These are proposed future extensions; they are not implemented in the current notebook.

---

# 👨‍💻 Author

**Rajan Kumar**

B.Tech – Computer Science & Engineering

---

## ⭐ Project Summary

This project demonstrates a complete **EDA and preprocessing workflow** on a heart disease dataset, including data inspection, statistical analysis, data-quality checks, target distribution analysis, correlation analysis, categorical encoding, and numerical feature standardization.
