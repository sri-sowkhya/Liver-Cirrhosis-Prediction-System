import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import pickle
import os

os.makedirs("../Flask", exist_ok=True)

df = pd.read_csv(r"C:\Users\bhagy\LiverCirrohsisPrediction\Data\archive (6) (1).csv")

df["Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)"] = (
    df["Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)"]
    .astype(str).str.strip().str.lower()
    .map({"yes": 1, "no": 0})
)

print("Raw class balance:")
print(df["Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)"].value_counts())

df['Gender'] = df['Gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
df['Obesity'] = df['Obesity'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
df['Diabetes'] = df['Diabetes Result'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

df.rename(columns={
    "Duration of alcohol consumption(years)": "AlcoholDuration",
    "Quantity of alcohol consumption (quarters/day)": "AlcoholQuantity",
    "Hemoglobin  (g/dl)": "Hemoglobin",
    "SGPT/ALT (U/L)": "SGPT"
}, inplace=True)

df.drop(columns=["S.NO", "Place(location where the patient lives)"], inplace=True, errors='ignore')

selected_features = [
    "Age", "Gender", "AlcoholDuration", "AlcoholQuantity", "Diabetes",
    "Obesity", "TCH", "LDL", "Hemoglobin", "SGPT"
]
target_column = "Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)"

for col in selected_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

df.dropna(subset=[target_column], inplace=True)

print("Class balance after cleaning:")
print(df[target_column].value_counts())

df_majority = df[df[target_column] == 1.0]
df_minority = df[df[target_column] == 0.0]

df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority),
    random_state=42
)

df_balanced = pd.concat([df_majority_downsampled, df_minority])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Class balance after undersampling:")
print(df_balanced[target_column].value_counts())

X = df_balanced[selected_features]
y = df_balanced[target_column]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

print("Class balance after SMOTE:")
print(pd.Series(y_balanced).value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

with open("../Flask/rf_acc_68.pkl", "wb") as f:
    pickle.dump(model, f)
with open("../Flask/normalizer.pkl", "wb") as f:
    pickle.dump(scaler, f)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))