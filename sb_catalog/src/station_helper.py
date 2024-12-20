"""
Parse stations from csv file and upload to the MongoDB
"""

import argparse
from pathlib import Path

import pandas as pd

from .utils import SeisBenchDatabase


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
    db = SeisBenchDatabase(args.db_uri, args.database)
    db.write_stations(stations)


if __name__ == "__main__":
    main()
