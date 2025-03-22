from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from retriever import fetch_and_process_data 

data = fetch_and_process_data(season='2023-24', season_type='Regular Season', games_to_process=10)

features = ['Home_Lead', 'Lead_Change', 'Score_Change', 'Time_Since_Last_Score']
labels = 'Momentum_Shift'

X = data[features]
y = data[labels]

X = X.dropna()
y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))
