import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv("data/digital_advertisement_dataset.csv")

X = df.drop("performance_class", axis=1)
y = df["performance_class"]

categorical_features = [
    "platform", "ad_type", "target_audience"
]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

joblib.dump(model, "digital_advertisement_performance_model.pkl")

ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Digital Advertisement Performance - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nModel saved as digital_advertisement_performance_model.pkl")
