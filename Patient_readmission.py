# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC 1. Import libraries

# COMMAND ----------

import pandas as pd
import numpy as np


# COMMAND ----------

# MAGIC %md
# MAGIC 2. Load the dataset

# COMMAND ----------

# Load dataset
df = spark.table("patients.readmission.dataset").toPandas()

# Display dataset shape
print("Dataset Shape:", df.shape)

# Display first five rows
display(df.head())

#Displaying the exact column names
print("Column names:")
print(df.columns.tolist())

# Check data types
print(df.dtypes)

# Summary statistics for numeric columns
display(df.describe())

# Summary statistics for categorical columns
display(df.describe(include="object"))

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Misiing colums/values

# COMMAND ----------

missing_audit = pd.DataFrame({
    "Missing count": df.isna().sum(),
    "Missing percent": (df.isna().mean() * 100).round(2)})
display(missing_audit)

# checking for duplicate rows and ids
duplicate_rows = df.duplicated().sum()
duplicate_ids = df["Patient ID"].duplicated().sum()

print("Duplicate rows:", duplicate_rows)
print("Duplicate Patient IDs:", duplicate_ids)

# COMMAND ----------

categorical_columns = ["Gender", "Admission Type", "Readmission"]

for col in categorical_columns:
    print(f"\\n{col} – unique values:")
    print(df[col].dropna().unique())
    print(df[col].value_counts(dropna=False))

print("\\nWhitespace audit:")
for col in categorical_columns:
    non_string = df[col].dropna().map(lambda value: not isinstance(value, str)).sum()
    padded = df[col].dropna().astype(str).ne(df[col].dropna().astype(str).str.strip()).sum()
    print(f"{col}: non-string values={non_string}, values with outer spaces={padded}")

# COMMAND ----------

expected_ranges = {
    "Patient ID": (1, 3000),
    "Age": (18, 89),
    "Length of Stay": (1, 29),
    "Number of Diagnoses": (1, 9),
    "Blood Pressure": (60, 179),
    "Blood Sugar Levels": (51, 199),
    "Previous Admissions": (0, 4)
}

range_rows = []
for col, (lower, upper) in expected_ranges.items():
    invalid_mask = ~df[col].between(lower, upper)
    range_rows.append({
        "Column": col,
        "Expected minimum": lower,
        "Expected maximum": upper,
        "Observed minimum": df[col].min(),
        "Observed maximum": df[col].max(),
        "Invalid count": int(invalid_mask.sum())
    })

range_audit = pd.DataFrame(range_rows)
display(range_audit)

# COMMAND ----------

numeric_audit_columns = [
    "Age", "Length of Stay", "Number of Diagnoses",
    "Blood Pressure", "Blood Sugar Levels", "Previous Admissions"
]

iqr_rows = []
for col in numeric_audit_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    iqr_rows.append({
        "Column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
        "Lower fence": lower, "Upper fence": upper,
        "Outlier count": int(count)})

iqr_audit = pd.DataFrame(iqr_rows)
display(iqr_audit)

# COMMAND ----------

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, numeric_audit_columns):
    sns.boxplot(x=df_raw[col], ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

# COMMAND ----------

df = df.copy()

# Rule 1: remove accidental leading/trailing spaces in categorical labels
for col in categorical_columns:
    df[col] = df[col].astype("string").str.strip()

# Rule 2: values outside the case-study data dictionary ranges are invalid;
# mark them as missing so they can be imputed using training data only.
for col, (lower, upper) in expected_ranges.items():
    df.loc[~df[col].between(lower, upper), col] = np.nan

print("Shape after audit-based cleaning:", df.shape)
print("Missing values after invalid-range checks:")
display(df.isna().sum().to_frame("Missing count"))

# COMMAND ----------

gender_map = {"Female": 0, "Male": 1}
admission_map = {"Elective": 0, "Emergency": 1}
target_map = {"No": 0, "Yes": 1}

df["Gender"] = df["Gender"].map(gender_map)
df["Admission Type"] = df["Admission Type"].map(admission_map)
y = df["Readmission"].map(target_map)

if y.isna().any():
    raise ValueError("Unexpected or missing Readmission target labels. Investigate before modelling.")

print("Encoded predictor examples:")
display(df[["Gender", "Admission Type"]].head())
print("\nEncoded target distribution:")
display(y.value_counts().sort_index().rename(index={0: "No (0)", 1: "Yes (1)"}))

# COMMAND ----------

feature_columns = [
    "Age", "Gender", "Admission Type", "Length of Stay",
    "Number of Diagnoses", "Blood Pressure",
    "Blood Sugar Levels", "Previous Admissions"
]

X = df[feature_columns].copy()

print("Predictor shape:", X.shape)
print("Target shape:", y.shape)
print("Predictor columns:", X.columns.tolist())

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

def class_balance(target):
    return pd.DataFrame({
        "Count": target.value_counts().sort_index(),
        "Percentage": (target.value_counts(normalize=True).sort_index() * 100).round(2)
    }).rename(index={0: "No (0)", 1: "Yes (1)"})

print("Training shape:", X_train.shape, " Test shape:", X_test.shape)
print("\nTraining target balance:")
display(class_balance(y_train))
print("\nTest target balance:")
display(class_balance(y_test))

# COMMAND ----------

numeric_features = [
    "Age", "Length of Stay", "Number of Diagnoses",
    "Blood Pressure", "Blood Sugar Levels", "Previous Admissions"
]
binary_features = ["Gender", "Admission Type"]

numeric_imputer = SimpleImputer(strategy="median")
binary_imputer = SimpleImputer(strategy="most_frequent")

X_train_num_imputed = pd.DataFrame(
    numeric_imputer.fit_transform(X_train[numeric_features]),
    columns=numeric_features, index=X_train.index)
X_test_num_imputed = pd.DataFrame(
    numeric_imputer.transform(X_test[numeric_features]),
    columns=numeric_features, index=X_test.index)

X_train_bin_imputed = pd.DataFrame(
    binary_imputer.fit_transform(X_train[binary_features]),
    columns=binary_features, index=X_train.index)
X_test_bin_imputed = pd.DataFrame(
    binary_imputer.transform(X_test[binary_features]),
    columns=binary_features, index=X_test.index)

X_train_imputed = pd.concat([X_train_num_imputed, X_train_bin_imputed], axis=1)[feature_columns]
X_test_imputed = pd.concat([X_test_num_imputed, X_test_bin_imputed], axis=1)[feature_columns]

print("Missing values after training-fitted imputation – train:", int(X_train_imputed.isna().sum().sum()))
print("Missing values after training-fitted imputation – test:", int(X_test_imputed.isna().sum().sum()))

# COMMAND ----------

scaler = StandardScaler()

X_train_scaled_num = pd.DataFrame(
    scaler.fit_transform(X_train_imputed[numeric_features]),
    columns=numeric_features, index=X_train_imputed.index
)
X_test_scaled_num = pd.DataFrame(
    scaler.transform(X_test_imputed[numeric_features]),
    columns=numeric_features, index=X_test_imputed.index
)

X_train_ready = pd.concat(
    [X_train_scaled_num, X_train_imputed[binary_features]], axis=1
)[feature_columns]
X_test_ready = pd.concat(
    [X_test_scaled_num, X_test_imputed[binary_features]], axis=1
)[feature_columns]

display(X_train_ready.head())
print("Training numeric means (approximately 0):")
display(X_train_ready[numeric_features].mean().round(3))
print("Training numeric standard deviations (approximately 1):")
display(X_train_ready[numeric_features].std(ddof=0).round(3))

# COMMAND ----------

before_summary = {
    "Rows": len(df),
    "Columns": df.shape[1],
    "Missing cells": int(df.isna().sum().sum()),
    "Duplicate rows": int(df.duplicated().sum()),
    "Duplicate Patient IDs": int(df["Patient ID"].duplicated().sum())
}
after_summary = {
    "Rows": len(df),
    "Columns": df.shape[1],
    "Missing cells after range validation": int(df.isna().sum().sum()),
    "Duplicate rows": int(df.duplicated().sum()),
    "Duplicate Patient IDs": int(df["Patient ID"].duplicated().sum())
}

summary = pd.DataFrame(
    {"Before cleaning": before_summary, "After audit-based cleaning": after_summary}
)
display(summary)

print("Training rows:", len(X_train_ready))
print("Testing rows:", len(X_test_ready))
print("Training target counts:", y_train.value_counts().sort_index().to_dict())
print("Testing target counts:", y_test.value_counts().sort_index().to_dict())