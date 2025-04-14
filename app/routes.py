from flask import render_template, request, jsonify
from app.models.train_model import train_model
from app.models.evaluate_model import evaluate_model
from app.models.save_model import save_model
from app.predictions.predict import make_predictions
import pandas as pd
from app.config.settings import MODEL_PATH

def configure_routes(app):

    @app.route('/')
    def home():
        return render_template('train_model.html')

    @app.route('/train', methods=['POST'])
    def train():
        from app.data.load_data import load_data
        raw_dat = load_data()
        
        from app.data.preprocess import preprocess_data
        X_train, X_test, y_train, y_test, X, y = preprocess_data(raw_dat)

        model = train_model(X_train, y_train)

        results = evaluate_model(model, X_test, y_test)
        print("Evaluation results:", results)

        save_model(model, MODEL_PATH)
        return jsonify(results), 200

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()

        new_data = pd.DataFrame([data['cdd_vals']], columns=[f'cdd{i+1}' for i in range(len(data['cdd_vals']))])

        from app.predictions.predict import load_model
        model = load_model()

        predictions, probabilities = make_predictions(model, new_data)

        results = {
            'predicted_ust': int(predictions[0]),
            'probability_ust': float(f"{probabilities[0]:.4f}")
        }

        return jsonify(results), 200
