import os
import joblib
import json
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, Model, IntegerField,
    FloatField, TextField, BooleanField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import numpy as np
import re
from data_cleaning import clean_data  # Importando a função clean_data
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

########################################
# Begin database stuff

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = BooleanField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Define data cleaning functions

def clean_c_charge_degree(degree):
    return re.sub(r'[^a-zA-Z]', '', degree)

def extract_year(date):
    return pd.to_datetime(date).year

def extract_month(date):
    return pd.to_datetime(date).month

def process_dates(df):
    df['dob_year'] = df['dob'].apply(extract_year)
    df['c_jail_year'] = df['c_jail_in'].apply(extract_year)
    df['c_jail_month'] = df['c_jail_in'].apply(extract_month)
    return df.drop(columns=['dob', 'c_jail_in'])

def agrupar_tipo_crime(descricao):
    if pd.isna(descricao):
        return 'other'
    descricao = descricao.lower()
    if 'battery' in descricao or 'assault' in descricao or 'violence' in descricao or 'murder' in descricao or 'batt' in descricao:
        return 'violence'
    elif 'theft' in descricao or 'burglary' in descricao or 'robbery' in descricao:
        return 'robbery'
    elif 'drug' in descricao or 'possession' in descricao or 'trafficking' in descricao or 'poss' in descricao or 'cocaine' in descricao or 'heroin' in descricao or 'deliver' in descricao or 'traffick' in descricao:
        return 'drugs'
    elif 'driving' in descricao or 'traffic' in descricao or 'license' in descricao or 'driv' in descricao or 'vehicle' in descricao or 'conduct' in descricao:
        return 'traffic'
    else:
        return 'other'

def group_races(df):
    race_map = df['race'].value_counts()
    common_races = race_map[race_map >= 50].index.tolist()
    df['race_grouped'] = df['race'].apply(lambda x: x if x in common_races else 'Other')
    return df.drop(columns=['race'])

def clean_data(df):
    # Drop columns
    df = df.drop(columns=['id', 'name', 'c_case_number', 'c_offense_date', 'c_arrest_date'])
    
    # Apply custom transformations
    df['c_charge_degree'] = df['c_charge_degree'].apply(clean_c_charge_degree)
    df['c_charge_desc'] = df['c_charge_desc'].apply(agrupar_tipo_crime)
    # Group races
    df = group_races(df)
    
    # Convert to categorical
    df['c_charge_desc'] = pd.Categorical(df['c_charge_desc'], categories=['violence', 'robbery', 'drugs', 'traffic', 'other'])
    df['sex'] = df['sex'].astype('category')
    df['race_grouped'] = df['race_grouped'].astype('category')
    df['c_charge_degree'] = df['c_charge_degree'].astype('category')
    
    # Process dates
    df = process_dates(df)
    
    return df

# End data cleaning functions
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)


with open('dtypes.pkl', 'rb') as fh:
    dtypes = pickle.load(fh)


pipeline = joblib.load('pipeline.pkl')


# End model un-pickling
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)

@app.route('/will_recidivate/', methods=['POST'])
def will_recidivate():
    request_data = request.json

    # Função para verificar se um valor está dentro de um intervalo
    def is_within_range(value, min_value, max_value):
        return min_value <= value <= max_value

    # Função para validar campos numéricos
    def validate_numerical_field(value, field_name, min_value, max_value):
        if pd.isna(value):
            return None  # Permitir valores NaN
        if not isinstance(value, (int, float)):
            return f"Invalid value '{value}' for '{field_name}'"
        elif not is_within_range(value, min_value, max_value):
            return f"Value '{value}' for '{field_name}' out of range [{min_value}, {max_value}]"
        else:
            return None

    # Função para validar campos categóricos
    def validate_categorical_field(value, field_name, valid_values):
        if pd.isna(value):
            return None  # Permitir valores NaN
        if value not in valid_values:
            return f"Invalid value '{value}' for '{field_name}'"
        return None

    # Função para validar campos de data/hora
    def validate_datetime(value, field_name):
        if pd.isna(value):
            return None  # Permitir valores NaN
        try:
            pd.to_datetime(value)
            return None
        except ValueError:
            return f"Invalid datetime format for '{field_name}'"

    response = {}

    observation_id = request_data.get("id", None)
    if observation_id is None:
        response["error"] = "Missing observation_id"
        return jsonify(response), 400

    # Verificar se o ID já existe no banco de dados
    if Prediction.select().where(Prediction.observation_id == observation_id).exists():
        response["error"] = "Observation ID already exists"
        return jsonify(response), 400

    # Validar campos
    data = request_data
    required_fields = ["id", "name", "sex", "dob", "race", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count", "c_case_number", "c_charge_degree", "c_charge_desc", "c_offense_date", "c_arrest_date", "c_jail_in"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        response["error"] = f"Missing fields: {', '.join(missing_fields)}"
        return jsonify(response), 400

    # Validar campos categóricos
    valid_categories = {
        'sex': ['Male', 'Female'],
        'c_charge_degree': ['F', 'M'],
        'race': ['Caucasian', 'African-American', 'Other', 'Hispanic', 'Native American', 'Asian']
    }
    
    for feature, valid_values in valid_categories.items():
        if feature in data:
            error_message = validate_categorical_field(data[feature], feature, valid_values)
            if error_message:
                response["error"] = error_message
                return jsonify(response), 400

    # Validar campos numéricos
    numerical_features = {"juv_fel_count": (0, 50), "juv_misd_count": (0, 50), "juv_other_count": (0, 50), "priors_count": (0, 50)}  
    for feature, (min_value, max_value) in numerical_features.items():
        if feature in data:
            error_message = validate_numerical_field(data[feature], feature, min_value, max_value)
            if error_message:
                response["error"] = error_message
                return jsonify(response), 400

    # Validar campos de data/hora
    datetime_features = ["dob", "c_offense_date", "c_arrest_date", "c_jail_in"]
    for feature in datetime_features:
        if feature in data:
            error_message = validate_datetime(data[feature], feature)
            if error_message:
                response["error"] = error_message
                return jsonify(response), 400

    # Converter os dados de entrada para DataFrame e ajustar os tipos de dados
    try:
        # Aplicar transformações do pipeline
        df = pd.DataFrame([data])
        df = clean_data(df)
        probability = pipeline.predict_proba(df)[0][1]
    
        # Aplicar o threshold de 0.6
        threshold = 0.6
        prediction = probability >= threshold

        # Map prediction to True or False
        prediction_label = bool(prediction)

        # Salvar a previsão no banco de dados
        Prediction.create(
            observation_id=observation_id,
            observation=json.dumps(data),
            proba=probability,
            true_class=None
        )

        # Construct response dictionary
        response = {
            "id": observation_id,
            "outcome": prediction_label,
        }
        return jsonify(response), 200
    except Exception as e:
        response["error"] = str(e)
        return jsonify(response), 500

@app.route('/recidivism_result/', methods=['POST'])
def recidivism_result():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['outcome']
        p.save()
        response = {
            "id": p.observation_id,
            "outcome": p.true_class,
            "predicted_outcome": p.proba >= 0.6
        }
        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = f'Observation ID: {obs["id"]} does not exist'
        return jsonify({'error': error_msg})

# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)