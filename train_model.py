import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score

tf.keras.utils.set_random_seed(42)

# ---------- 1. Load and clean data ----------
df = pd.read_csv("student_data.csv")
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))
print("Dataset shape:", df.shape)

# ---------- 2. Split and scale ----------
features = ["Attendance", "Assignment_Marks", "Internal_Score",
            "Study_Hours", "Previous_GPA"]
X = df[features].values
y = df["At_Risk"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- 3. Class weights ----------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# ---------- 4. Build model ----------
model = keras.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---------- 5. Train ----------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ---------- 6. Evaluate ----------
y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
print(classification_report(y_test, y_pred,
                            target_names=["Not At Risk", "At Risk"]))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

# ---------- 7. Save ----------
model.save("student_performance_model.keras")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully!")