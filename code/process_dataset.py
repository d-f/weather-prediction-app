import numpy as np
import utils
from pathlib import Path
from typing import Dict, List
import argparse


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-raw_data_filepath", default=Path("C:\\personal_ML\\weather_prediction_app\\raw_dataset.json"))
    parser.add_argument("-save_filepath", default=Path("C:\\personal_ML\\weather_prediction_app\\normalized_dataset.json"))
    return parser.parse_args()


def find_max(raw_data: Dict, data_name: str) -> int:
    max_value = 0
    for _, data_dict in raw_data.items():
        data = data_dict[data_name]
        for value in data:
            if int(value) > max_value:
                max_value = int(value)
    return max_value


def find_min(raw_data: Dict, data_name: str) -> int:
    min_value = np.inf
    for _, data_dict in raw_data.items():
        data = data_dict[data_name]
        for value in data:
            if int(value) < min_value:
                min_value = int(value)
    return min_value


def get_maxes(raw_data: Dict) -> Dict:
    max_vis = find_max(raw_data=raw_data, data_name="VISIBILITY")
    max_wb = find_max(raw_data=raw_data, data_name="WET_BULB_TEMP")
    max_ws = find_max(raw_data=raw_data, data_name="WIND_SPEED")
    return {
        "VISIBILITY": max_vis,
        "WET_BULB_TEMP": max_wb,
        "WIND_SPEED": max_ws
    }


def get_mins(raw_data: Dict) -> Dict:
    min_vis = find_min(raw_data=raw_data, data_name="VISIBILITY")
    min_wb = find_min(raw_data=raw_data, data_name="WET_BULB_TEMP")
    min_ws = find_min(raw_data=raw_data, data_name="WIND_SPEED")
    return {
        "VISIBILITY": min_vis,
        "WET_BULB_TEMP": min_wb,
        "WIND_SPEED": min_ws
    }


def normalize(min_val: int, max_val: int, x: int) -> float:
    range_val = max_val - min_val
    x -= min_val
    x /= range_val
    return x


def normalize_data(raw_data: Dict, mins: Dict, maxes: Dict, data_names: List) -> Dict:
    norm_data = {x: {} for x in raw_data.keys()}
    for data_name in data_names:
        for station, data_dict in raw_data.items():
            norm_data_list = [normalize(min_val=mins[data_name], max_val=maxes[data_name], x=int(x)) for x in data_dict[data_name]]

            norm_data[station].update({data_name: norm_data_list})
            del data_dict[data_name]
            norm_data[station].update(data_dict)
    return norm_data


def main():
    args = create_argparser()
    raw_data = utils.read_json(args.raw_data_filepath)
    maxes = get_maxes(raw_data=raw_data)
    mins = get_mins(raw_data=raw_data)
    norm_data = normalize_data(raw_data=raw_data, mins=mins, maxes=maxes, data_names=["VISIBILITY", "WET_BULB_TEMP", "WIND_SPEED"])
    utils.save_json(json_dict=norm_data, file_path=Path(args.save_filepath))


if __name__ == "__main__":
    main()
