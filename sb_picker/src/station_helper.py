"""
Parse stations from text file
"""

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
from picker import S3MongoSBBridge, SeisBenchCollection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--db_uri", type=str, required=False)
    parser.add_argument("--collection", type=str, required=False)
    args = parser.parse_args()

    stations = get_stations(args.path)
    if args.db_uri is None:
        import IPython

        IPython.embed()
    else:
        write_stations(stations, args.db_uri, args.collection)


def write_stations(stations: pd.DataFrame, db_uri: str, collection: str) -> None:
    db = SeisBenchCollection(db_uri, collection)
    bridge = S3MongoSBBridge(s3=None, db=db)

    bridge._insert_many_ignore_duplicates("stations", stations.to_dict("records"))


def get_stations(path: Path) -> pd.DataFrame:
    sub_dfs = []
    with open(path, "r") as f:
        full_text = f.read()

    full_text = full_text.split("\n\n")
    header = full_text[0].strip()[1:].replace(" ", "")

    for block in full_text[1:]:
        parts = block.split("\n")

        datacenter, url = parts[0][1:].split(",")
        datacenter = datacenter[11:]
        data = "\n".join([header] + parts[1:])
        df = pd.read_csv(StringIO(data), sep="|")
        df["datacenter"] = datacenter
        df["datacenter_url"] = url
        sub_dfs.append(df)

    df = pd.concat(sub_dfs)

    df.rename(
        columns={
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Elevation": "Elevation",
            "Network": "network_code",
            "Station": "station_code",
            "Sitename": "sitename",
            "StartTime": "starttime",
            "EndTime": "endtime",
        },
        inplace=True,
    )

    return df


if __name__ == "__main__":
    main()
