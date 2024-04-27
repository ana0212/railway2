import os
import joblib
import json
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
from sklearn.linear_model import LogisticRegression


########################################
# Begin database stuff

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class BinaryLogisticRegression(LogisticRegression):
    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)


with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


pipeline = joblib.load('pipeline.pickle')


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.json
    
    # Function to check if a value is within a given range
    def is_within_range(value, min_value, max_value):
        return min_value <= value <= max_value

    # Function to validate numerical fields
    def validate_numerical_field(value, field_name, min_value, max_value):
        if not isinstance(value, (int, float)):
            return f"Invalid value '{value}' for '{field_name}'"
        elif not is_within_range(value, min_value, max_value):
            return f"Value '{value}' for '{field_name}' out of range [{min_value}, {max_value}]"
        else:
            return None

    response = {}
    
    observation_id = request_data.get("observation_id", None)
    if observation_id is None:
        response["error"] = "Missing observation_id"
        return jsonify(response)
    
    data = request_data.get("data", None)
    if data is None:
        response["error"] = "Missing 'data' field in request"
        return jsonify(response)

    # Check if all required fields are present
    required_fields = ['age', 'sex', 'race', 'workclass', 'education', 'marital-status',
                       'capital-gain', 'capital-loss', 'hours-per-week']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        response["error"] = f"Missing fields: {', '.join(missing_fields)}"
        return jsonify(response)

    # Check for extra fields
    extra_fields = [field for field in data if field not in required_fields]
    if extra_fields:
        response["error"] = f"Extra fields: {', '.join(extra_fields)}"
        return jsonify(response)

    # Validate categorical features
    valid_categories = {
        "sex": ["Male", "Female"],
        "race": ["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"],
        "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"],
        "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
        "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
    }
    
    for feature, valid_values in valid_categories.items():
        if feature in data and data[feature] not in valid_values:
            response["error"] = f"Invalid value '{data[feature]}' for '{feature}'"
            return jsonify(response)

    # Validate numerical features
    numerical_features = {'age': (0, 100), 'capital-gain': (0, 99999), 'capital-loss': (0, 99999), 'hours-per-week': (0, 168)}  # Based on observed values
    for feature, (min_value, max_value) in numerical_features.items():
        if feature in data:
            error_message = validate_numerical_field(data[feature], feature, min_value, max_value)
            if error_message:
                response["error"] = error_message
                return jsonify(response)

    # Perform prediction if request is properly structured
    try:
        # Extract data and make prediction
        df = pd.DataFrame([data])
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0][1]

        # Map prediction to True or False
        prediction_label = bool(prediction)

        # Construct response dictionary
        response = {
            "observation_id": observation_id,
            "prediction": prediction_label,
            "probability": probability
        }
        return jsonify(response)
    except Exception as e:
        response["error"] = str(e)
        return jsonify(response)

@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
