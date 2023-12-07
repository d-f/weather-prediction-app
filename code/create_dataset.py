import json
from pathlib import Path
import pymongo 
import argparse
from tqdm import tqdm
import utils


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-host_name", default="localhost:27017/")
    parser.add_argument("-db_name", default="noaa_global_marine")
    parser.add_argument("-col_name", default="wp_app")
    parser.add_argument("-save_path", default="C:\\personal_ML\\weather_prediction_app\\raw_dataset.json")
    return parser.parse_args()


def get_client(host_name: str):
    client = pymongo.MongoClient(f"mongodb://{host_name}")
    return client


def get_database(client, database_name: str):
    return client[database_name]


def get_collection(database, collection_name: str):
    return database[collection_name]


def get_stations(collection):
    return collection.distinct("STATION")


def create_dataset(collection, stations):
    dataset = {x: {"WET_BULB_TEMP":[],"VISIBILITY":[],"WIND_SPEED":[],"DATE":[]} for x in stations}
    for station_name in tqdm(dataset.keys()):
        for doc in collection.find(
            {
            "WET_BULB_TEMP": {"$ne": ""}, 
            "VISIBILITY": {"$ne": ""},
            "WIND_SPEED": {"$ne": ""},
            "STATION": station_name
            }
            ).sort("DATE", 1):
            dataset[station_name]["WET_BULB_TEMP"].append(doc["WET_BULB_TEMP"])
            dataset[station_name]["VISIBILITY"].append(doc["VISIBILITY"])
            dataset[station_name]["WIND_SPEED"].append(doc["WIND_SPEED"])
            dataset[station_name]["DATE"].append(doc["DATE"])
    dataset = {x: y for x, y in dataset.items() if len(y["WET_BULB_TEMP"]) > 0}
    return dataset


def save_json(json_dict, file_path: Path):
    with open(file_path, mode="w") as opened_json:
        json.dump(json_dict, opened_json)    


def main():
    args = create_argparser()
    client = utils.get_client(host_name=args.host_name)
    database = utils.get_database(client=client, database_name=args.db_name)
    collection = utils.get_collection(database=database, collection_name=args.col_name)
    stations = get_stations(collection=collection)
    station_dict = create_dataset(collection=collection, stations=stations)
    save_json(json_dict=station_dict, file_path=Path(args.save_path))


if __name__ == "__main__":
    main()
