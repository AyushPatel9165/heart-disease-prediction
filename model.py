import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

def train_model():
    
    data = pd.read_csv(r"C:\Data Science\Project Assignments and Project Topic list\Heart Disease Predicition\heart.csv")
    data.columns = data.columns.str.strip()
 
    X = data.drop("target", axis=1)
    y = data["target"]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

 
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    
    pipeline.fit(X_train, y_train)

    
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Model Accuracy:", round(acc * 100, 2), "%")
    print("Confusion Matrix:\n", cm)

    
    with open("heart_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("Model Saved Successfully!")

if __name__ == "__main__":
    train_model()