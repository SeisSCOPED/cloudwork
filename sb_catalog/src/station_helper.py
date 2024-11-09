"""
Parse stations from csv file and
initialize the MongoDB / DocumentDB
"""

import argparse
from pathlib import Path

import pandas as pd

from .util import SeisBenchDatabase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=Path, help="Path to CSV that contians station metadata."
    )
    parser.add_argument(
        "--db_uri", type=str, required=True, help="URI of the MongoDB cluster."
    )
    parser.add_argument(
        "--database", type=str, default="tutorial", help="MongoDB database name."
    )
    args = parser.parse_args()

    stations = pd.read_csv(args.path)
    write_stations(stations, args.db_uri, args.database)


def write_stations(stations: pd.DataFrame, db_uri: str, database: str) -> None:
    db = SeisBenchDatabase(db_uri, database)

    db.insert_many_ignore_duplicates("stations", stations.to_dict("records"))


if __name__ == "__main__":
    main()
