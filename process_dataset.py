import math
from datetime import datetime
import utils
from pathlib import Path
from typing import Dict
import argparse


def parse_argparser() -> argparse.Namespace:
    """parses command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-raw_data_filepath", type=Path)
    parser.add_argument("-n_save_filepath", type=Path)
    parser.add_argument("-val_prop", type=float)
    parser.add_argument("-input_width", type=int)
    parser.add_argument("-label_width", type=int)
    parser.add_argument("-data_prep_json", type=Path)
    return parser.parse_args()


def partition_dataset(dataset: Dict, val_prop: float) -> Dict:
    """
    partition dataset into train, validation and test datasets

    keyword arguments:
    dataset  -- dictionary with timestamp keys and temperature values
    val_prop -- proportion of the amount of the end of the sequence
        to reserve for validation and testing
    """
    dataset_size = len(dataset)
    val_amt = int(dataset_size*val_prop)
    part_dataset = {"train": {}, "val": {}, "test": {}}

    dataset_list = [(x, y) for x, y in dataset.items()]

    train_ts = dataset_list[0:dataset_size-(val_amt*2)] 
    val_ts = dataset_list[dataset_size-(val_amt*2):dataset_size-val_amt]
    test_ts = dataset_list[dataset_size-val_amt:dataset_size]
    part_dataset["train"].update({x: y for x, y in dataset.items() if (x, y) in train_ts})
    part_dataset["val"].update({x: y for x, y in dataset.items() if (x, y) in val_ts})
    part_dataset["test"].update({x: y for x, y in dataset.items() if (x, y) in test_ts})

    return part_dataset
    

def window_sequence(
        num_windows: int, 
        total_size: int, 
        dataset: Dict, 
        window_dataset: Dict, 
        input_width: int, 
        label_width: int, 
        mode: str
        ) -> Dict:
    """
    breaks the sequence up into data windows with
    consecutive inputs and outputs

    keyword arguments:
    num_windows    -- number of total windows in the dataset partition
    total_size     -- size of the input and output vector combined
    dataset        -- dictionary of train, val and test dictionaries which have input and output pairs
    window_dataset -- dictionary containing windowed dataset partitions
    input_width    -- size of the input vector
    label_width    -- size of the output vector
    mode           -- partition name {"train", "val", "test"}
    """
    for window_idx in range(num_windows):
        if utils.data_utils.check_window_duration(
            start=datetime.fromtimestamp(
                datetime.timestamp(list(dataset["train"].keys())[window_idx*total_size])
            ),
            end=datetime.fromtimestamp(
                datetime.timestamp(list(dataset["train"].keys())[(window_idx*total_size)+total_size])
            ),
            total_size=total_size
        ):
            total_vector = list(dataset["train"].values())[window_idx*total_size:(window_idx*total_size)+total_size]
            window_dataset[mode]["input"].append(total_vector[:input_width])
            window_dataset[mode]["output"].append(total_vector[-label_width:])
    return window_dataset        


def create_window_dataset(dataset: Dict, input_width: int, label_width: int) -> Dict:
    """
    creates a dataset of data windows

    keyword arguments:
    dataset     -- dictionary of train, val and test dictionaries which have input and output pairs
    input_width -- size of the input vector
    label_width -- size of the output vector
    """
    window_dataset = {
        "train": {"input": [], "output": []},
        "val": {"input": [], "output": []},
        "test": {"input": [], "output": []}
    }
    total_size = input_width+label_width
    num_train_windows = math.floor(len(dataset["train"]) / total_size)
    num_val_windows = math.floor(len(dataset["val"]) / total_size)
    num_test_windows = math.floor(len(dataset["test"]) / total_size)

    window_sizes = [num_train_windows, num_val_windows, num_test_windows]
    dataset_modes = ["train", "val", "test"]
    for num_windows, mode in zip(window_sizes, dataset_modes):
        window_dataset.update(window_sequence(
            num_windows=num_windows,
            total_size=total_size,
            dataset=dataset,
            window_dataset=window_dataset,
            input_width=input_width, 
            label_width=label_width,
            mode=mode
        ))
    return window_dataset


def main():
    args = parse_argparser()
    raw_data = utils.data_utils.read_json(args.raw_data_filepath)
    raw_data = utils.data_utils.convert_keys(raw_data)
    raw_series = utils.data_utils.dict_to_series(raw_data)
    missing_time_idxs = utils.data_utils.find_missing_times(raw_series)
    raw_series = utils.data_utils.fill_in_missing(raw_series=raw_series, missing_times=missing_time_idxs)
    utils.data_utils.save_data_prep(raw_series=raw_series, file_path=args.data_prep_json)
    norm_data = utils.data_utils.normalize_dataset(
        raw_series=raw_series, 
        min_val=raw_series.min(), 
        max_val=raw_series.max()
    )
    partitioned_n_dataset = partition_dataset(dataset=norm_data, val_prop=args.val_prop)
    windowed_n_dataset = create_window_dataset(
        dataset=partitioned_n_dataset, 
        input_width=args.input_width, 
        label_width=args.label_width
    )
    utils.data_utils.save_json(json_dict=windowed_n_dataset, file_path=Path(args.n_save_filepath))    


if __name__ == "__main__":
    main()
