from datetime import datetime

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator


JOBS_DIR = "/usr/local/airflow/jobs"
POSTGRES_JAR = "/opt/spark/jars/postgresql-42.7.0.jar"
POSTGRES_CONN_ID = "pg_covid_bd"

RAW_TABLE = "raw.covid_deaths_raw"
STAGE_TABLE = "stage.covid_deaths_daily"
MART_TABLE = "mart.covid_deaths_monthly"

CONNECTION_CREDS = {
    "POSTGRES_USER": f"{{{{ conn.{POSTGRES_CONN_ID}.login }}}}",
    "POSTGRES_PASSWORD": f"{{{{ conn.{POSTGRES_CONN_ID}.password }}}}",
    "POSTGRES_HOST": f"{{{{ conn.{POSTGRES_CONN_ID}.host }}}}",
    "POSTGRES_PORT": f"{{{{ conn.{POSTGRES_CONN_ID}.port }}}}",
    "POSTGRES_DB": f"{{{{ conn.{POSTGRES_CONN_ID}.schema }}}}",
}

COMMON_ENV_VARS = {
    "DATA_PATH": "/usr/local/airflow/data",
    "PYTHONPATH": JOBS_DIR,
    **CONNECTION_CREDS,
}


with DAG(
    dag_id="dag_build_covid_bd",
    description="ETL pipeline for COVID deaths data: RAW -> STAGE -> MART",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["covid", "spark", "etl"],
) as dag:
    job_raw_layer = SparkSubmitOperator(
        task_id="job_raw_layer",
        conn_id="spark_default",
        application=f"{JOBS_DIR}/job_raw_layer.py",
        jars=POSTGRES_JAR,
        name="job_raw_layer",
        verbose=True,
        env_vars=COMMON_ENV_VARS,
        application_args=["--raw-table", RAW_TABLE],
    )

    job_stage_layer = SparkSubmitOperator(
        task_id="job_stage_layer",
        conn_id="spark_default",
        application=f"{JOBS_DIR}/job_stage_layer.py",
        jars=POSTGRES_JAR,
        name="job_stage_layer",
        verbose=True,
        env_vars=COMMON_ENV_VARS,
        application_args=["--raw-table", RAW_TABLE, "--stage-table", STAGE_TABLE],
    )

    job_mart_layer = SparkSubmitOperator(
        task_id="job_mart_layer",
        conn_id="spark_default",
        application=f"{JOBS_DIR}/job_mart_layer.py",
        jars=POSTGRES_JAR,
        name="job_mart_layer",
        verbose=True,
        env_vars=COMMON_ENV_VARS,
        application_args=["--stage-table", STAGE_TABLE, "--mart-table", MART_TABLE],
    )

    job_raw_layer >> job_stage_layer >> job_mart_layer
