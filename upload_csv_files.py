from pathlib import Path
import argparse 
import utils


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-host_name")
    parser.add_argument("-db_name")
    parser.add_argument("-col_name")
    parser.add_argument("-csv_dir", type=Path)
    return parser.parse_args()
    

def upload_dir(csv_dir: Path, collection):
    for csv_path in csv_dir.iterdir():
        dataset_csv = utils.data_utils.read_csv(csv_path)
        ds_json_list = utils.data_utils.csv_to_json(opened_csv=dataset_csv)
        insert_result = collection.insert_many(ds_json_list)
        print(insert_result)


def main():
    args = create_argparser()
    client = utils.data_utils.mongo_client(host_name=args.host_name)
    database = utils.data_utils.mongo_database(client=client, database_name=args.db_name)
    collection = utils.data_utils.mongo_collection(database=database, collection_name=args.col_name)
    upload_dir(csv_dir=args.csv_dir, collection=collection)


if __name__ == "__main__":
    main()
