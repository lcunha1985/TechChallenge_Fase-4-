import warnings; warnings.filterwarnings("ignore")
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import shap
import os

OBESITY_ORDER = [
    "Insufficient_Weight","Normal_Weight",
    "Overweight_Level_I","Overweight_Level_II",
    "Obesity_Type_I","Obesity_Type_II","Obesity_Type_III",
]
OBESITY_LABELS = {
    "Insufficient_Weight":"Abaixo do Peso","Normal_Weight":"Peso Normal",
    "Overweight_Level_I":"Sobrepeso I","Overweight_Level_II":"Sobrepeso II",
    "Obesity_Type_I":"Obesidade I","Obesity_Type_II":"Obesidade II",
    "Obesity_Type_III":"Obesidade III",
}
ALL_FEATURES = [
    "Gender","Age","Height","Weight","BMI",
    "family_history","FAVC","FCVC","NCP","CAEC",
    "SMOKE","CH2O","SCC","FAF","TUE","CALC","MTRANS",
]
CAT_COLS = ["Gender","family_history","FAVC","CAEC","SMOKE","SCC","CALC","MTRANS"]

print("Carregando dados...")
df = pd.read_csv("data/Obesity.csv")
df["BMI"] = (df["Weight"] / (df["Height"] ** 2)).round(2)
df["Obesity_Label"] = df["Obesity"].map(OBESITY_LABELS)
for col in ["FCVC","NCP","CH2O","FAF","TUE"]:
    df[col] = df[col].round().astype(int)
df["Obesity"] = pd.Categorical(df["Obesity"], categories=OBESITY_ORDER, ordered=True)

print("Encodando features...")
le_dict = {}
df_enc  = df.copy()
for col in CAT_COLS:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

le_target = LabelEncoder()
le_target.fit(OBESITY_ORDER)
df_enc["target"] = le_target.transform(df["Obesity"].astype(str))

X = df_enc[ALL_FEATURES]
y = df_enc["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Treinando XGBoost...")
model = XGBClassifier(
    n_estimators=300, learning_rate=0.08, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="mlogloss", verbosity=0)
model.fit(X_train, y_train)

acc_test = accuracy_score(y_test, model.predict(X_test))
print(f"Acurácia teste: {acc_test:.4f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
print(f"Acurácia CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("Calculando SHAP values...")
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_test[:300])

os.makedirs("models", exist_ok=True)

print("Salvando modelo...")
with open("models/model.pkl", "wb") as f:
    pickle.dump(dict(
        model=model,
        le_dict=le_dict,
        le_target=le_target,
        X_test=X_test,
        y_test=y_test,
        acc_test=acc_test,
        cv_scores=cv_scores,
        shap_vals=shap_vals,
        X_shap=X_test[:300],
    ), f)

print("Modelo salvo em models/model.pkl")
print("Pronto!")