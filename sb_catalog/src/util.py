import datetime
import logging
from typing import Any

import pandas as pd
import pymongo
from pymongo.errors import BulkWriteError, DuplicateKeyError
from pymongo.results import InsertManyResult

logger = logging.getLogger("sb_picker")


class SeisBenchDatabase(pymongo.MongoClient):
    """
    A MongoDB Client designed to handle all necessary tables for creating a simple earthquake catalog.
    It provides useful helper functions and a structure.
    """

    def __init__(self, db_uri: str, database: str, **kwargs: Any) -> None:
        super().__init__(db_uri, **kwargs)

        self.db_uri = db_uri
        self.database = super().__getitem__(database)

        self.colls = {"picks", "stations", "sb_runs", "events", "assignments"}
        self._setup()

    def _setup(self) -> None:
        """
        Setup indices for the main tables for faster access.
        Tables are generally created lazily.
        """
        pick_coll = self.database["picks"]
        if "unique_index" not in pick_coll.index_information():
            pick_coll.create_index(
                ["trace_id", "phase", "time"], unique=True, name="unique_index"
            )

        station_coll = self.database["stations"]
        if "station_idx" not in station_coll.index_information():
            station_coll.create_index(["id"], unique=True, name="station_idx")

    def get_stations(self, extent: tuple[float, float, float, float]) -> pd.DataFrame:
        """
        Returns a DataFrame with all stations within the given range.
        """
        minlat, maxlat, minlon, maxlon = extent

        cursor = self.database["stations"].find(
            {
                "latitude": {"$gt": minlat, "$lt": maxlat},
                "longitude": {"$gt": minlon, "$lt": maxlon},
            }
        )

        return pd.DataFrame(list(cursor))

    def insert_many_ignore_duplicates(
        self, key: str, entries: list[dict[str, Any]]
    ) -> InsertManyResult:
        """
        Inserts many keys into a table while ignoring any duplicates.
        All other errors in inserting the data are passed to the user.
        """
        try:
            return self.database[key].insert_many(
                entries,
                ordered=False,  # Not ordered to make sure every query is sent
            )
        except DuplicateKeyError:
            logger.warning(
                f"Some duplicate entries have been skipped while writing to collection '{key}'."
            )
        except BulkWriteError as e:
            # See https://www.mongodb.com/docs/manual/reference/error-codes/ for full error code
            if all(x["code"] == 11000 for x in e.details["writeErrors"]):
                logger.warning("Some duplicate entries have been skipped.")
            else:
                raise e


def parse_year_day(x: str) -> datetime.date:
    return datetime.datetime.strptime(x, "%Y.%j").date()


def s3_path_mapper(net, sta, loc, cha, year, day, c) -> str:
    try:
        s3 = network_mapper[net]
    except KeyError:
        raise NotImplementedError(f"Network {net} not implemented. Check src.util.")
    prefix = _prefix_mapper(s3, net, year, day)
    basename = _basename_mapper(s3, net, sta, loc, cha, year, day, c)
    return f"{prefix}{basename}"


def _prefix_mapper(s3, net, year, day) -> str:
    if s3 == "ncedc-pds":
        return f"{s3}/continuous_waveforms/{net}/{year}/{year}.{day}/"
    elif s3 == "scedc-pds":
        return f"{s3}/continuous_waveforms/{year}/{year}_{day}/"


def _basename_mapper(s3, net, sta, loc, cha, year, day, c) -> str:
    if s3 == "ncedc-pds":
        return f"{sta}.{net}.{cha}{c}.{loc}.D.{year}.{day}"
    elif s3 == "scedc-pds":
        return f"{net}{sta.ljust(5, '_')}{cha}{c}{loc.ljust(3, '_')}{year}{day}.ms"


network_mapper = {
    "AZ": "scedc-pds",
    "CI": "scedc-pds",
    "BK": "ncedc-pds",
    "CC": "ncedc-pds",
    "CE": "ncedc-pds",
    "GM": "ncedc-pds",
    "GS": "ncedc-pds",
    "NC": "ncedc-pds",
    "NN": "ncedc-pds",
    "NP": "ncedc-pds",
    "PB": "ncedc-pds",
    "PG": "ncedc-pds",
    "RE": "ncedc-pds",
    "SB": "ncedc-pds",
    "SF": "ncedc-pds",
    "TA": "ncedc-pds",
    "UO": "ncedc-pds",
    "US": "ncedc-pds",
    "UW": "ncedc-pds",
    "WR": "ncedc-pds",
}
