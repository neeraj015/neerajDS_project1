##my instructions

import dagshub
dagshub.init(repo_owner='neeraj015', repo_name='neerajDS_project1', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)