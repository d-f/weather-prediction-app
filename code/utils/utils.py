import pymongo
from pathlib import Path
import json
import csv


def get_client(host_name: str):
    client = pymongo.MongoClient(f"mongodb://{host_name}")
    return client


def get_database(client, database_name: str):
    return client[database_name]


def get_collection(database, collection_name: str):
    return database[collection_name]


def read_csv(csv_path: Path):
    with open(csv_path) as opened_file:
        reader = csv.reader(opened_file)
        return [x for x in reader]
    

def save_json(json_dict, file_path: Path):
    with open(file_path, mode="w") as opened_json:
        json.dump(json_dict, opened_json) 
        

def read_json(json_path: Path):
    with open(json_path) as opened_json:
        return json.load(opened_json)
