import statsmodels.api as sm
import pandas as pd
import math
from datetime import datetime, timedelta
import numpy as np
import utils
from pathlib import Path
from typing import Dict, List, Iterable
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-raw_data_filepath", default=Path("C:\\personal_ML\\weather_prediction_app\\raw_dataset.json"))
    parser.add_argument("-s_save_filepath", default=Path("C:\\personal_ML\\weather_prediction_app\\standardized_dataset.json"))
    parser.add_argument("-n_save_filepath", default=Path("C:\\personal_ML\\weather_prediction_app\\normalized_dataset.json"))
    parser.add_argument("-plot", action="store_true", default=True)
    parser.add_argument("-val_prop", type=float, default=0.1)
    parser.add_argument("-input_width", type=int, default=20)
    parser.add_argument("-label_width", default=1)
    return parser.parse_args()


def normalize(min_val: int, max_val: int, x: int) -> float:
    range_val = max_val - min_val
    x -= min_val
    x /= range_val
    return x


def standardize(mean, std, x):
    x -= mean
    x /= std
    return x


def normalize_dataset(seasonal_stationary: Dict, max_val: int, min_val: int) -> Dict:
    return {ts: normalize(x=tmp, min_val=min_val, max_val=max_val) for ts, tmp in seasonal_stationary.items()}


def standardize_dataset(seasonal_stationary, mean, std):
    return {ts: standardize(x=tmp, mean=mean, std=std) for ts, tmp in seasonal_stationary.items()}


def model_seasonality(raw_series) -> List:
    print("")
    result = sm.tsa.seasonal_decompose(raw_series)
    return result





def plot_seasonal_model(raw_data, seasonal_model):
    plt.close()
    plt.plot(raw_data.keys(), raw_data.values(), color="blue")
    plt.plot(raw_data.keys(), seasonal_model, color="orange")
    plt.show()
 

def plot_seasonal_stable(raw_data, seasonal_model):
    diff = np.array(list(raw_data.values())) - np.array(seasonal_model)
    plt.close()
    plt.plot(raw_data.keys(), diff)
    plt.show()


# def remove_seasonality(raw_data, seasonal_model):
#     diff = np.array(list(raw_data.values())) - np.array(seasonal_model)
#     return {x: y for x, y in zip(raw_data.keys(), diff)}


# def get_seasonal_model(raw_data, coeffs):
#     return [get_sm_value(x=x, coeffs=coeffs) for x in raw_data.values()]


def partition_dataset(dataset, val_prop):
    dataset_size = len(list(dataset.keys()))
    val_amt = int(dataset_size*val_prop)
    part_dataset = {"train": {}, "val": {}, "test": {}}
    ts_list = list(dataset.keys())
    train_ts = ts_list[0:dataset_size-(val_amt*2)] 
    val_ts = ts_list[dataset_size-(val_amt*2):dataset_size-val_amt]
    test_ts = ts_list[dataset_size-val_amt:dataset_size]
    part_dataset["train"].update({x: y for x, y in dataset.items() if x in train_ts})
    part_dataset["val"].update({x: y for x, y in dataset.items() if x in val_ts})
    part_dataset["test"].update({x: y for x, y in dataset.items() if x in test_ts})

    return part_dataset


def check_window_duration(start, end, total_size):
    if end - start == timedelta(minutes=5*total_size):
        return True
    else:
        return False
    

def window_sequence(num_windows, total_size, dataset, window_dataset, input_width, label_width):
    for window_idx in range(num_windows):
        if check_window_duration(
            start=datetime.fromtimestamp(list(dataset["train"].keys())[window_idx*total_size]),
            end=datetime.fromtimestamp(list(dataset["train"].keys())[(window_idx*total_size)+total_size]),
            total_size=total_size
        ):
            total_vector = list(dataset["train"].values())[window_idx*total_size:(window_idx*total_size)+total_size]
            window_dataset["train"]["input"].append(total_vector[:input_width])
            window_dataset["train"]["output"].append(total_vector[-label_width])
    return window_dataset        


def create_window_dataset(dataset, input_width, label_width):
    window_dataset = {
        "train": {"input": [], "output": []},
        "val": {"input": [], "output": []},
        "test": {"input": [], "output": []}
    }
    total_size = input_width+label_width
    num_train_windows = math.floor(len(dataset["train"]) / total_size)
    num_val_windows = math.floor(len(dataset["val"]) / total_size)
    num_test_windows = math.floor(len(dataset["test"]) / total_size)

    for num_windows in [num_train_windows, num_val_windows, num_test_windows]:
        window_dataset = window_sequence(
            num_windows=num_windows,
            total_size=total_size,
            dataset=dataset,
            window_dataset=window_dataset,
            input_width=input_width, 
            label_width=label_width
        )
    return window_dataset


def convert_keys(raw_dict):
    return {float(x): y for x, y in raw_dict.items()}


def dict_to_series(raw_dict):
    return pd.Series(data=raw_dict.values(), index=[datetime.fromtimestamp(x) for x in raw_dict.keys()])


def main():
    args = create_argparser()
    raw_data = utils.read_json(args.raw_data_filepath)
    raw_data = convert_keys(raw_data)
    raw_series = dict_to_series(raw_data)
    model_seasonality(raw_series=raw_series)
    # seasonal_model = get_seasonal_model(raw_data=raw_data, coeffs=rev_coeffs)
    # plot_seasonal_model(raw_data=raw_data, seasonal_model=seasonal_model)
    # plot_seasonal_stable(raw_data=raw_data, seasonal_model=seasonal_model)
    # seasonal_stationary = remove_seasonality(raw_data=raw_data, seasonal_model=seasonal_model)
    norm_data = normalize_dataset(seasonal_stationary=seasonal_stationary, min_val=min_val, max_val=max_val)
    standard_data = standardize_dataset(seasonal_stationary=seasonal_stationary, mean=mean, std=std)
    partitioned_n_dataset = partition_dataset(dataset=norm_data, val_prop=args.val_prop)
    partitioned_s_dataset = partition_dataset(dataset=standard_data, val_prop=args.val_prop)
    windowed_n_dataset = create_window_dataset(dataset=partitioned_n_dataset, input_width=args.input_width, label_width=args.label_width)
    windowed_s_dataset = create_window_dataset(dataset=partitioned_s_dataset, input_width=args.input_width, label_width=args.label_width)
    utils.save_json(json_dict=windowed_n_dataset, file_path=Path(args.n_save_filepath))
    utils.save_json(json_dict=windowed_s_dataset, file_path=Path(args.s_save_filepath))

    # fix seasonality not doing anything


if __name__ == "__main__":
    main()
