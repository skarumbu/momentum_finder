import argparse
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from retriever import fetch_and_process_data


def get_current_season() -> str:
    now = datetime.now()
    if now.month >= 10:
        return f"{now.year}-{str(now.year + 1)[2:]}"
    else:
        return f"{now.year - 1}-{str(now.year)[2:]}"


parser = argparse.ArgumentParser()
parser.add_argument('--season', default=None, help='Season in YYYY-YY format (default: current season)')
parser.add_argument('--games', type=int, default=None, help='Max games to train on (default: all games this season)')
args = parser.parse_args()

season = args.season or get_current_season()
print(f"Training on season {season}, games={args.games or 'all'}...")

data = fetch_and_process_data(season=season, season_type='Regular Season', games_to_process=args.games)

features = ['Home_Lead', 'Lead_Change', 'Score_Change', 'Time_Since_Last_Score']
labels = 'Momentum_Shift'

X = data[features].dropna()
y = data[labels][X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

joblib.dump(model, "momentum_model.pkl")
print("Model saved to momentum_model.pkl")
