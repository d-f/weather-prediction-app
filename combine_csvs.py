import csv 
import argparse
from pathlib import Path
import utils


def parse_argparser():
    """parses command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv_dir", type=Path)
    parser.add_argument("-save_path", type=Path)
    return parser.parse_args()


def gather_all_csvs(csv_dir: Path) -> list:
    """combines all result csv files together to upload to SQL database"""
    combined = []
    for csv_path in csv_dir.glob("*.csv"):
        tmp_csv = utils.read_csv(csv_path)[1:][0]
        combined.append(
            (
                tmp_csv[0], 
                int(tmp_csv[1]), 
                float(tmp_csv[2]), 
                int(tmp_csv[3]), 
                int(tmp_csv[4]),
                int(tmp_csv[5]),
                int(tmp_csv[6]),
                float(tmp_csv[7])
            )
            )
    return combined


def save_csv(combined: list, file_path: Path) -> None:
    """saves csv file"""
    with open(file_path, mode="w", newline="") as opened_csv:
        writer = csv.writer(opened_csv)
        for row in combined:
            writer.writerow(row)


def main():
    args = parse_argparser()
    combined = gather_all_csvs(csv_dir=args.csv_dir)
    save_csv(combined=combined, file_path=args.save_path)


if __name__ == "__main__":
    main()
