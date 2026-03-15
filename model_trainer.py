import argparse
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from azure.storage.blob import BlobServiceClient
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

MIN_ROC_AUC = 0.60

X = data[FEATURES].dropna()

all_valid = True
for label, output_path in [(HOME_LABEL, 'home_run_model.pkl'), (AWAY_LABEL, 'away_run_model.pkl')]:
    y = data[label][X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n--- {label} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(classification_report(y_test, y_pred))

    if auc < MIN_ROC_AUC:
        print(f"ERROR: {label} ROC-AUC {auc:.4f} is below minimum {MIN_ROC_AUC}. Aborting upload.")
        all_valid = False
    else:
        joblib.dump(model, output_path)
        print(f"Saved to {output_path}")

if not all_valid:
    raise SystemExit(1)

# Upload models to Azure Blob Storage so the server can download them on cold start.
conn_str = os.getenv('MODEL_STORAGE_CONNECTION_STRING')
if conn_str:
    container = BlobServiceClient.from_connection_string(conn_str).get_container_client('models')
    for blob_name in ('home_run_model.pkl', 'away_run_model.pkl'):
        if os.path.exists(blob_name):
            with open(blob_name, 'rb') as f:
                container.upload_blob(blob_name, f, overwrite=True)
            print(f"Uploaded {blob_name} to blob storage")
else:
    print("MODEL_STORAGE_CONNECTION_STRING not set; skipping blob upload")
