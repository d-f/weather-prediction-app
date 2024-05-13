import pymongo
from pathlib import Path
import json
import csv
from typing import Dict, List
import pandas as pd
import datetime
import numpy as np


def csv_to_json(opened_csv: List) -> List:
    """
    converts each csv row to a json dict
    and appends to json_list

    keyword arguments
    opened_csv -- opened csv list from read_csv() function
    """
    json_list = []
    headers = opened_csv[0]
    for row in opened_csv[1:]:
        json_list.append(
            {x: y for x, y in zip(headers, row)}
            )
    return json_list


def mongo_client(host_name: str) -> pymongo.MongoClient:
    """
    returns MongoDB client

    keyword arguments:
    host_name -- string of the name of the host to 
        add to mongodb client connection string
    """
    client = pymongo.MongoClient(f"mongodb://{host_name}")
    return client


def mongo_database(
        client: pymongo.MongoClient, 
        database_name: str
        ) -> pymongo.MongoClient:
    """
    returns the database from the client

    keyword arguments:
    client        -- MongoDB client
    database_name -- name of the database
    """
    return client[database_name]


def mongo_collection(database: pymongo.database, collection_name: str) -> pymongo.database:
    """
    returns the collection from the database

    keyword arguments:
    database        -- pymongo database
    collection_name -- name of the collection
    """
    return database[collection_name]


def dataset_dates(collection: pymongo.collection):
    """
    returns all discinct time values in the dataset

    keyword arguments:
    collection -- pymongo collection
    """
    return collection.distinct("DATE")


def read_csv(csv_path: Path) -> list:
    """
    reads csv file and returns list of rows

    keyword arguments:
    csv_path: path to the .csv file 
    """
    with open(csv_path) as opened_file:
        reader = csv.reader(opened_file)
        return [x for x in reader]
    

def save_json(json_dict: Dict, file_path: Path) -> None:
    """
    saves json file

    keyword arguments:
    json_dict -- dictionary object to JSON serialize
    file_path -- path of the JSON file to save
    """
    with open(file_path, mode="w") as opened_json:
        json.dump(json_dict, opened_json) 
        

def read_json(json_path: Path) -> Dict:
    """
    opens json file

    keyword arguments:
    json_path -- path of the JSON file to read
    """
    with open(json_path) as opened_json:
        return json.load(opened_json)


def save_train_results(train_loss: list, val_loss: list, result_fp: Path) -> None:
    """
    saves the train and validation loss per epoch

    keyword arguments:
    train_loss -- list of tuples (epoch, loss_value) on train dataset per epoch
    val_loss -- list of tuples (epoch, loss_value) on val dataset per epoch
    result_fp -- filepath to save results to
    """
    save_dict = {
        "train loss per epoch": train_loss,
        "validation loss per epoch": val_loss
        }
    save_obj = json.dumps(save_dict)
    with open(result_fp, mode="w") as opened_json:
        opened_json.write(save_obj)


def save_experiment(
        result_fp: Path, 
        batch_size: int, 
        learning_rate: float, 
        num_layers: int, 
        hidden_size: int, 
        input_size: int, 
        output_size: int, 
        test_loss: float, 
        model_save_name: str
        ) -> None:
    """
    saves the test performance, model name and hyperparameters

    keyword arguments:
    result_fp       -- filepath to save experiment settings 
    batch_size      -- batch size used during training
    learning_rate   -- learning rate used for optimizer during training
    num_layers      -- number of recurrent LSTM layers 
    hidden_size     -- number of features in each hidden state
    input_size      -- size of the input vector
    output_size     -- size of the prediction vector
    test_loss       -- mean squared error on test dataset
    model_save_name -- model name, including .pth.tar
    """
    csv_list = [
        ("model_save_name", "batch_size", "lr", "num_layers", "hidden_size", "input_size", "output_size", "test_loss"),
        (model_save_name, batch_size, learning_rate, num_layers, hidden_size, input_size, output_size, test_loss)
    ]
    with open(result_fp, mode="w", newline="") as opened_csv:
        writer = csv.writer(opened_csv)
        for row in csv_list:
            writer.writerow(row)


def normalize(min_val: int, max_val: int, x: int) -> float:
    """
    normalizes value between 0 and 1 according to
    minimum and maximum of dataset

    keyword arguments:
    min_val -- maxium value from dataset
    max_val -- minimum value from dataset
    x       -- value to normalze
    """
    range_val = max_val - min_val
    x -= min_val
    x /= range_val
    return x


def normalize_dataset(raw_series: type[pd.Series], min_val: float, max_val: float):
    """
    applies normalize() function to all values within a series

    keyword arguments:
    raw_series -- pandas series
    min_val    -- maximum of pandas series
    max_val    -- minimum of pandas series
    """
    return {timestamp: normalize(x=temp, min_val=min_val, max_val=max_val) for timestamp, temp in raw_series.items()}


def convert_keys(raw_dict: Dict) -> Dict:
    """convert dictionary key data types to floats"""
    return {float(x): y for x, y in raw_dict.items()}


def dict_to_series(raw_dict: Dict) -> Dict:
    """convert dictionary to pandas series"""
    return pd.Series(data=raw_dict.values(), index=[datetime.datetime.fromtimestamp(x) for x in raw_dict.keys()])


def find_missing_times(raw_series: pd.Series) -> list:
    """finds which positions in the series are missing"""
    missing_time_idxs = []
    rs_list = [x for x in raw_series.keys()]
    for time_entry_idx in range(len(raw_series.keys())-1):
        if not time_entry_idx+1 > len(raw_series.keys()):
            diff = rs_list[time_entry_idx+1] - rs_list[time_entry_idx]
            if diff > pd.Timedelta(minutes=5):
                num_skipped_period = diff / pd.Timedelta(minutes=5)
                for period_idx in range(int(num_skipped_period-1)): # -1 in order to not replace the end of the missing window
                    missing_time_idxs.append(
                        (
                        rs_list[time_entry_idx]+((period_idx+1)*pd.Timedelta(minutes=5))
                         )
                        )
    return missing_time_idxs


def fill_in_missing(raw_series: pd.Series, missing_times: list) -> pd.Series:
    """
    fills in the missing time values with linear interpolation

    keyword arguments:
    raw_series    -- pandas series with missing values
    missing_times -- index values of missing times
    """
    for timestamp in missing_times:
        idx = [x for x in raw_series.keys()].index(timestamp-pd.Timedelta(minutes=5))
        s1 = raw_series[0:idx+1]
        s2 = pd.Series({timestamp: np.nan})
        s3 = raw_series[idx+1:]
        raw_series = pd.concat([s1, s2, s3])

    raw_series = raw_series.interpolate()
    return raw_series


def save_data_prep(raw_series: pd.Series, file_path: Path) -> None:
    """saves minimum and maximum values for inference normalization"""
    save_dict = {"min": raw_series.min(), "max": raw_series.max()}
    with open(file_path, mode="w") as opened_json:
        json.dump(save_dict, opened_json)


def check_window_duration(start, end, total_size, num_minutes):
    """
    checks whether the difference between two sizes
    is equal to total_size
    """
    if end - start == datetime.timedelta(minutes=num_minutes*total_size):
        return True
    else:
        return False
    
