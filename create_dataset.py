from datetime import datetime
from pathlib import Path
import pymongo 
import argparse
from tqdm import tqdm
import utils
from typing import Type


def parser_argparser():
    """return command-line argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-host_name", type=str, default="localhost:27017/")
    parser.add_argument("-db_name", type=str, default="noaa_global_hourly")
    parser.add_argument("-col_name", type=str, default="wp_app")
    parser.add_argument("-save_path", type=Path)
    return parser.parse_args()


def create_dataset(collection: Type[pymongo.collection], dates: list):
    """
    creates dataset from pymongo collection

    keyword arguments:
    collection -- pymongo collection
    dates      -- list result from dataset_dates() in /utils/data_utils.py
    """
    dataset = {}
    for date in tqdm(dates):
        for doc in collection.find(
            # get all documents with ,1 in the TMP measurement and a TMP != +9999
            {
            "DATE": date,
            "TMP": {"$regex": ",1", "$nin": ["+9999"]}
            }
        ):
            year = int(doc["DATE"].partition("-")[0])
            month = int(doc["DATE"].partition("-")[2].partition("-")[0])
            day = int(doc["DATE"].partition("-")[2].partition("-")[2].partition("T")[0])
            hour = int(doc["DATE"].partition("T")[2].partition(":")[0])
            minute = int(doc["DATE"].rpartition(":")[0].rpartition(":")[2])
            dt = datetime(year, month, day, hour, minute)
            dt.replace(hour=hour, minute=minute)
            temp = int(doc["TMP"].partition(",")[0][1:])
            if "-" == doc["TMP"][0]:
                temp *= -1
            dataset[dt.timestamp()] = temp

    return dataset


def main():
    args = parser_argparser()
    client = utils.data_utils.mongo_client(host_name=args.host_name)
    database = utils.data_utils.mongo_database(client=client, database_name=args.db_name)
    collection = utils.data_utils.mongo_collection(database=database, collection_name=args.col_name)
    dates = utils.data_utils.dataset_dates(collection=collection)
    station_dict = create_dataset(collection=collection, dates=dates)
    utils.data_utils.save_json(json_dict=station_dict, file_path=Path(args.save_path))


if __name__ == "__main__":
    main()
