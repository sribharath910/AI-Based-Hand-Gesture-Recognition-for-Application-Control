import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load datasets
df_swipe_right = pd.read_csv("swipe_right.csv")
df_swipe_right["label"] = 0

df_swipe_left = pd.read_csv("swipe_left.csv")
df_swipe_left["label"] = 1

df_stop = pd.read_csv("playstop.csv")
df_stop["label"] = 2

df_volume_up = pd.read_csv("volume_up.csv")
df_volume_up["label"] = 3

df_volume_down = pd.read_csv("volume_down.csv")
df_volume_down["label"] = 4

df_swipe_up = pd.read_csv("swipe_up.csv")
df_swipe_up["label"] = 5

df_swipe_down = pd.read_csv("swipe_down.csv")
df_swipe_down["label"] = 6

# Combine all data
df = pd.concat([
    df_swipe_right, df_swipe_left, df_stop,
    df_volume_up, df_volume_down,
    df_swipe_up, df_swipe_down
], ignore_index=True)

# X: all 63 features (hand landmarks), y: labels
X = df.iloc[:, :-1].values
y = df["label"].values

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "gesture_model.pkl")
print("✅ Model saved as gesture_model.pkl")
