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
