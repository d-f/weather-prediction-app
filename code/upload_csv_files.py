from typing import List
from pathlib import Path
import csv 
import argparse 
import pymongo


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-host_name", default="localhost:27017")
    parser.add_argument("-db_name", default="noaa_global_marine")
    parser.add_argument("-col_name", default="wp_app")
    parser.add_argument("-csv_dir", default="C:\\personal_ML\\weather_prediction_app\\noaa_data\\", type=Path)
    return parser.parse_args()


def read_csv(csv_path: Path):
    with open(csv_path) as opened_file:
        reader = csv.reader(opened_file)
        return [x for x in reader]
    

def csv_to_json(opened_csv: List):
    json_list = []
    headers = opened_csv[0]
    for row in opened_csv[1:]:
        json_list.append(
            {x: y for x, y in zip(headers, row)}
            )
    return json_list


def get_client(host_name: str):
    client = pymongo.MongoClient(f"mongodb://{host_name}")
    return client


def get_database(client, database_name: str):
    return client[database_name]


def get_collection(database, collection_name: str):
    return database[collection_name]


def upload_dir(csv_dir: Path, collection):
    for csv_path in csv_dir.iterdir():
        dataset_csv = read_csv(csv_path)
        ds_json_list = csv_to_json(opened_csv=dataset_csv)
        insert_result = collection.insert_many(ds_json_list)
        print(insert_result)


def main():
    args = create_argparser()
    client = get_client(host_name=args.host_name)
    database = get_database(client=client, database_name=args.db_name)
    collection = get_collection(database=database, collection_name=args.col_name)
    upload_dir(csv_dir=args.csv_dir, collection=collection)


if __name__ == "__main__":
    main()
