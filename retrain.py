import numpy as np
import joblib
import json
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Load embeddings
print("Loading embeddings...")
train_data = np.load("train.npz")
val_data   = np.load("val.npz")
test_data  = np.load("test.npz")

img_embs_train = train_data['image_embs']
txt_embs_train = train_data['text_embs']
labels_train   = train_data['labels']

img_embs_val = val_data['image_embs']
txt_embs_val = val_data['text_embs']
labels_val   = val_data['labels']

img_embs_test = test_data['image_embs']
txt_embs_test = test_data['text_embs']
labels_test   = test_data['labels']

# Build features
def build_features(img_embs, txt_embs):
    cosine_sim = (img_embs * txt_embs).sum(axis=1, keepdims=True)
    return np.hstack([img_embs, txt_embs, cosine_sim])

X_train = build_features(img_embs_train, txt_embs_train)
X_val   = build_features(img_embs_val,   txt_embs_val)
X_test  = build_features(img_embs_test,  txt_embs_test)

y_train, y_val, y_test = labels_train, labels_val, labels_test

print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Train
print("Training MLP...")
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),  # smaller than (512, 256, 128)
    activation='relu',
    solver='adam',
    learning_rate_init=1e-3,
    max_iter=20,              # fewer iterations
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=3,       # stop sooner
    random_state=42,
    verbose=True
)
mlp.fit(X_train_scaled, y_train)

# Evaluate
print("\n--- Validation ---")
val_preds = mlp.predict(X_val_scaled)
val_probs = mlp.predict_proba(X_val_scaled)[:, 1]
print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")
print(f"F1:       {f1_score(y_val, val_preds):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_val, val_probs):.4f}")

print("\n--- Test ---")
test_preds = mlp.predict(X_test_scaled)
test_probs = mlp.predict_proba(X_test_scaled)[:, 1]
acc  = accuracy_score(y_test, test_preds)
f1   = f1_score(y_test, test_preds)
auc  = roc_auc_score(y_test, test_probs)
print(f"Accuracy: {acc:.4f}")
print(f"F1:       {f1:.4f}")
print(f"ROC-AUC:  {auc:.4f}")
print(classification_report(y_test, test_preds, target_names=['Real', 'Fake']))

# Save locally with current NumPy/Python version
joblib.dump(mlp, "classifier.pkl")
joblib.dump(scaler, "scaler.pkl")

# Update config
config = {
    "clip_model": "ViT-B/32",
    "embedding_dim": 512,
    "feature_dim": 1025,
    "labels": {0: "AUTHENTIC", 1: "MANIPULATED"},
    "label_colors": {0: "#00FF9C", 1: "#FF4B4B"},
    "test_accuracy": round(acc, 4),
    "test_f1": round(f1, 4),
    "test_roc_auc": round(auc, 4)
}
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\nSaved: classifier.pkl, scaler.pkl, config.json")