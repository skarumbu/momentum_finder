import argparse
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from retriever import fetch_and_process_data, FEATURES, HOME_LABEL, AWAY_LABEL


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

X = data[FEATURES].dropna()

for label, output_path in [(HOME_LABEL, 'home_run_model.pkl'), (AWAY_LABEL, 'away_run_model.pkl')]:
    y = data[label][X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n--- {label} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, output_path)
    print(f"Saved to {output_path}")
