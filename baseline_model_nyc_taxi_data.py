#!/usr/bin/env python
# coding: utf-8

# # Baseline model for batch monitoring example


import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnQuantileMetric, ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

from joblib import load, dump
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	quantilemetric float,
	num_drifted_columns integer,
	share_missing_values float
)
"""

reference_data = pd.read_parquet('reference.parquet')


mar_data = pd.read_parquet('green_tripdata_2024-03.parquet')

begin = datetime.datetime(2024, 3, 1, 0, 0)
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']


column_mapping = ColumnMapping(
    target=None,
    prediction=None,
    numerical_features=num_features,
    categorical_features=cat_features
)

quantile_report = Report(metrics = [
    ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)
   


def calculate_metrics_postgresql(curr):
    begin = datetime.datetime(2024, 3, 1, 0, 0)

    for i in range(31):

        current_data = mar_data[(mar_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i))) &
        (mar_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))]

        quantile_report.run(reference_data = reference_data, current_data = current_data,
        column_mapping=column_mapping)

        result = quantile_report.as_dict()
        
        columnquantilemetric = result['metrics'][0]['result']['current']['value']
        num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
        share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

        curr.execute(
            "insert into dummy_metrics(timestamp, quantilemetric, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
            (begin + datetime.timedelta(i), columnquantilemetric, num_drifted_columns, share_missing_values)
	)

    
    
def batch_monitoring():
	prep_db()
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		with conn.cursor() as curr:
			calculate_metrics_postgresql(curr)
			logging.info("data sent")

if __name__ == '__main__':
	batch_monitoring()

