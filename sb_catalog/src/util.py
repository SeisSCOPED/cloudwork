import logging
from typing import Any, Mapping

import pandas as pd
import pymongo
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, DuplicateKeyError
from pymongo.results import InsertManyResult

logger = logging.getLogger("sb_picker")


class SeisBenchCollection(pymongo.MongoClient):
    """
    A MongoDB Client designed to handle all necessary tables for creating a simple earthquake catalog.
    It provides useful helper functions and a structure.
    """

    def __init__(self, db_uri: str, collection: str, **kwargs: Any) -> None:
        super().__init__(db_uri, **kwargs)

        self.db_uri = db_uri
        self.collection = collection
        self._collection = super().__getitem__(collection)

        self.dbs = {"picks", "stations", "sb_runs", "events", "assignments"}
        self._setup()

    def _setup(self) -> None:
        """
        Setup indices for the main tables for faster access.
        Tables are generally created lazily.
        """
        pick_db = self["picks"]
        if "unique_index" not in pick_db.index_information():
            pick_db.create_index(
                ["trace_id", "phase", "time"], unique=True, name="unique_index"
            )

        station_db = self["stations"]
        if "station_idx" not in station_db.index_information():
            station_db.create_index(["id"], unique=True, name="station_idx")

    def __getitem__(self, item) -> Collection[Mapping[str, Any] | Any]:
        if item not in self.dbs:
            raise KeyError(
                f"Database '{item}' not available. Possible options are: "
                + ", ".join(self.dbs)
            )
        return self._collection[item]

    def get_stations(self, extent: tuple[float, float, float, float]) -> pd.DataFrame:
        """
        Returns a DataFrame with all stations within the given range.
        """
        minlat, maxlat, minlon, maxlon = extent

        cursor = self["stations"].find(
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
            return self[key].insert_many(
                entries,
                ordered=False,  # Not ordered to make sure every query is sent
            )
        except DuplicateKeyError:
            logger.warning(
                f"Some duplicate entries have been skipped while writing to DB '{key}'."
            )
        except BulkWriteError as e:
            if all(x["code"] == 11000 for x in e.details["writeErrors"]):
                logger.warning("Some duplicate entries have been skipped.")
            else:
                raise e
