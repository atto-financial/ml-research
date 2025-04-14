import joblib

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")
