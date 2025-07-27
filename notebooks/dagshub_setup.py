import mlflow
import dagshub


mlflow.set_tracking_uri('https://dagshub.com/KARAN-sys141/mini-mlops.mlflow')
dagshub.init(repo_owner='KARAN-sys141', repo_name='mini-mlops', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric_name',1)