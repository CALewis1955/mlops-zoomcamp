#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import argparse
import pyarrow
#import logging

#logging.basicConfig(level=logging.DEBUG)

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
# get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=str, required=True, help='Year of the data')
parser.add_argument('--month', type=str, required=True, help='Month of the data')
args = parser.parse_args()
year = args.year
month = args.month


url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'
categorical = ['PULocationID', 'DOLocationID']

def read_data(url):
    df = pd.read_parquet(url)
    #logging.debug(f'Read the NYC taxi data file for {month}, {year}. The url was {url}. \
    #    The data has {df.shape[0]:,} rows and {df.shape[1]:,} columns of data.')
    print(f'Read the NYC taxi data file for {month}, {year}. The url was {url}. \
        The data has {df.shape[0]:,} rows and {df.shape[1]:,} columns of data.')
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(df):
    dicts = df[categorical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict(X)
    print(f'The mean predicted trip duration is {y_pred.mean():.3f} minutes.')
    return y_pred


def create_results_dataframe(df, y_pred):
    df['ride_id'] = df.tpep_pickup_datetime.dt.strftime('%Y/%m') + '_' + df.index.astype(str)
    ride_ids = df.ride_id.to_list()
    ride_dict = dict(zip(ride_ids, y_pred))
    df_results = pd.DataFrame.from_dict(ride_dict, orient='index', columns=["DurationPrediction"])
    df_results.reset_index(inplace=True)
    df_results.rename(columns={'index': 'Ride_Id', 'DurationPrediction': 'DurationPrediction'}, inplace=True)
    return df_results

def save_results(df_results):
    df_results.to_parquet(
        f'predictions_{year}_{month}.parquet',
        engine='pyarrow',
        compression=None,
        index=False
    )

def run(url):
    df = read_data(url)
    y_pred = predict(df)
    df_results = create_results_dataframe(df, y_pred)
    save_results(df_results)
    
if __name__ == '__main__':
    run(url)