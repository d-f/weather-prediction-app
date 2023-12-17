from datetime import datetime
import json
from pathlib import Path
import pymongo 
import argparse
from tqdm import tqdm
import utils


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-host_name", default="localhost:27017/")
    parser.add_argument("-db_name", default="noaa_global_hourly")
    parser.add_argument("-col_name", default="wp_app")
    parser.add_argument("-save_path", default="C:\\personal_ML\\weather_prediction_app\\raw_dataset.json")
    parser.add_argument("-threshold", type=int, default=20)
    return parser.parse_args()


def get_client(host_name: str):
    client = pymongo.MongoClient(f"mongodb://{host_name}")
    return client


def get_database(client, database_name: str):
    return client[database_name]


def get_collection(database, collection_name: str):
    return database[collection_name]


def get_dates(collection):
    return collection.distinct("DATE")


def create_dataset(collection, dates):
    dataset = {}
    for date in tqdm(dates):
        for doc in collection.find(
            # get all documents with ,1 in the TMP measurement and a TMP != +9999
            {
            "DATE": date,
            "TMP": "/,1/",
            "TMP": {"$ne": "+9999"} 
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


def save_json(json_dict, file_path: Path):
    with open(file_path, mode="w") as opened_json:
        json.dump(json_dict, opened_json)    


def main():
    args = create_argparser()
    client = utils.get_client(host_name=args.host_name)
    database = utils.get_database(client=client, database_name=args.db_name)
    collection = utils.get_collection(database=database, collection_name=args.col_name)
    dates = get_dates(collection=collection)
    station_dict = create_dataset(collection=collection, dates=dates)
    save_json(json_dict=station_dict, file_path=Path(args.save_path))


if __name__ == "__main__":
    main()
