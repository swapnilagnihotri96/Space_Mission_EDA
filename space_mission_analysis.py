import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def analyze_space_missions():
    print("Loading Space Mission Dataset...")
    try:
        df = pd.read_csv("space_mission_data.csv")
    except FileNotFoundError:
        print("Data file space_mission_data.csv not found.")
        return

    print("Cleaning and engineering features...")
    df['Year'] = pd.to_datetime(df['Datum'], errors='coerce').dt.year
    df['Is_Success'] = (df['Status Mission'] == 'Success').astype(int)
    
    print(f"Total Launches Analyzed: {len(df)}")
    print(f"Overall Success Rate: {df['Is_Success'].mean() * 100:.2f}%")

    # Feature preparation
    features = ['Company Name', 'Status Rocket', 'Year']
    X = pd.get_dummies(df[features].dropna(), drop_first=True)
    y = df.loc[X.index, 'Is_Success']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Model Test Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")

if __name__ == "__main__":
    analyze_space_missions()
