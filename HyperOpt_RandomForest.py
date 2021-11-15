#libraries imported
import sys
import argparse
import pyspark
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import hyperopt
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from sklearn.model_selection import GridSearchCV
from urllib.parse import urlparse
import mlflow
import mlflow.pyfunc
import mlflow.sklearn

# MLflow Backend store SQLite
# remote_server_uri = "http://localhost:5000" # set to your server URI
# mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

mlflow.set_tracking_uri("sqlite:///mlruns.db") 
'''
 connects to a tracking URI. You can also set the 
 MLFLOW_TRACKING_URI environment variable to have MLflow 
 find a URI from there. In both cases, the URI can either
 be a HTTP/HTTPS URI for a remote server, a database connection
 string, or a local path to log data to a directory. 

 '''
#evaluation function
def eval_metrics(actual, pred):
    rmse=np.sqrt(mean_squared_error(actual, pred))
    mae= mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Load the data

data = pd.read_csv('IRIS.csv')
# encode the species column
le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])
# Split the data into training and test sets. (0.70, 0.30) split.
y_data = data['species']
x_data = data.drop(['species'], axis=1)
train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.3)
#Hyperopt runs for multiple models
mod = RandomForestClassifier (max_depth=100, max_features=10,
    n_estimators=100)

def objective(params):
    with mlflow.start_run(nested=True):
        max_depth, max_features, n_estimators = params
        max_depth, max_features, n_estimators = (int(max_depth), float(max_features),int(n_estimators))
        mod = RandomForestClassifier (max_depth=max_depth, max_features=max_features,
        n_estimators=n_estimators)
        # Log all of our training parameters for this run.
        mlparams = {
            'max_depth': str(max_depth),
            'max_features': str(max_features),
            'n_estimators': str(n_estimators),
        }
        mlflow.log_params(mlparams)

        mod.fit(train_x, train_y)
        preds = mod.predict(test_x)
        acc = cross_val_score(mod, train_x, train_y, scoring="accuracy").mean()
        # acc = accuracy_score(test_y, preds)
        #log metric
        mlflow.log_metric("accuracy", acc)
        #log model
        mlflow.sklearn.log_model(mod, "saved_models")

        return {"loss": -acc, "status": STATUS_OK, 'mod': mlparams}

search_space = [
    hp.uniform("max_depth", 5, 100),
    hp.uniform("max_features", 0.1, 1.0),
    hp.uniform("n_estimators", 2, 1000),
]
trials = Trials()
with mlflow.start_run() as run:
    best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=2,trials=trials)




#Best huperopt model selected and runned
print(trials.best_trial['result']['mod'])
def unpacking(max_depth, max_features, n_estimators):
    max_depth_1 = max_depth
    max_features_1 = max_features
    n_estimators_1 = n_estimators
    return int(max_depth_1), float(max_features_1), int(n_estimators_1)

with mlflow.start_run() as run:
    best = trials.best_trial['result']['mod']
    s_l = unpacking(**best)
    max_depth, max_features, n_estimators = s_l
    #log parameters
    mlflow.log_params(best)

    mod = RandomForestClassifier(max_depth=max_depth, max_features=max_features,
                                 n_estimators=n_estimators)
    mod.fit(train_x, train_y)
    preds = mod.predict(test_x)
    acc = cross_val_score(mod, train_x, train_y, scoring="accuracy").mean()
    # acc = accuracy_score(test_y, preds)
    # log metric
    mlflow.log_metric("accuracy", acc)
    # log model
    mlflow.sklearn.log_model(mod, "saved_models")
    best_run = run.info




#Model registered and version is created
print('Model path:',best_run.artifact_uri)
print('Model run id:',best_run.run_id)
#
# import time
# model_name = 'Iris_dataset'
# client = mlflow.tracking.MlflowClient()
# try:
#   client.create_registered_model(model_name)
# except Exception as e:
#   pass
#
# model_version = client.create_model_version(model_name, f"{best_run.artifact_uri}/saved_models", best_run.run_id)
# time.sleep(1)
# # client.update_model_version(model_name, model_version.version, stage = 'Staging', description= 'Current candidate')
#
#
#
#
# #Model is transitioned into stage of production
# client = MlflowClient()
# client.transition_model_version_stage(
#     name="Iris_dataset",
#     version=1,
#     stage="Staging"
# )
#
#
# #Versioned model loaded and prediction is made in locally
# model_name = "Iris_dataset"
# model_version = 1
#
# model = mlflow.pyfunc.load_model(
#     model_uri=f"models:/{model_name}/{model_version}"
# )
#
# prediction = model.predict(test_x)
# print(prediction)

# #!/usr/bin/env sh
#
# # Set environment variable for the tracking URL where the Model Registry resides
# export MLFLOW_TRACKING_URI=http://localhost:5000
#
# # Serve the production model from the model registry
# mlflow models serve -m "models:/sk-learn-random-forest-reg-model/Production"
