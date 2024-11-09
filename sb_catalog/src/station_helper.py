"""
Parse stations from csv file
"""

import argparse
from pathlib import Path

import pandas as pd

from .util import SeisBenchCollection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=Path, helper="Path to CSV that contians station metadata."
    )
    parser.add_argument(
        "--db_uri", type=str, required=True, helper="MongoDB server URI."
    )
    parser.add_argument(
        "--collection", type=str, default="tutorial", helper="MongoDB collection name."
    )
    args = parser.parse_args()

    stations = pd.read_csv(args.path)
    write_stations(stations, args.db_uri, args.collection)


def write_stations(stations: pd.DataFrame, db_uri: str, collection: str) -> None:
    db = SeisBenchCollection(db_uri, collection)

    db.insert_many_ignore_duplicates("stations", stations.to_dict("records"))


if __name__ == "__main__":
    main()
