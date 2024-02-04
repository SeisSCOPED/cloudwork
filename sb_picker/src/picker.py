import argparse
import asyncio
import datetime
import logging
from typing import Iterable, Optional

import numpy as np
import obspy
import pymongo
import seisbench
import seisbench.models as sbm
import seisbench.util as sbu
from pymongo.errors import DuplicateKeyError
from s3fs import S3FileSystem

logger = logging.getLogger("sb_picker")


def parse_year_day(x: str) -> datetime.date:
    return datetime.datetime.strptime(x, "%Y.%j").date()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3", type=str, required=True)
    parser.add_argument("--s3_format", type=str, default="ncedc")
    parser.add_argument("--db_uri", type=str, required=True)
    parser.add_argument(
        "--db_container", type=str, required=True, help="db.container for MongoDB"
    )
    parser.add_argument(
        "--stations",
        type=str,
        required=True,
        help="Stations (comma separated) in format NET.STA.LOC.CHA without component.",
    )
    parser.add_argument(
        "--start",
        type=parse_year_day,
        required=True,
        help="Format: YYYY.DDD (included)",
    )
    parser.add_argument(
        "--end",
        type=parse_year_day,
        required=True,
        help="Format: YYYY.DDD (not included)",
    )
    parser.add_argument("--components", type=str, default="ZNE12")
    parser.add_argument("--model", type=str, default="PhaseNet")
    parser.add_argument("--weight", type=str, default="instance")
    parser.add_argument("--p_threshold", default=0.2, type=float)
    parser.add_argument("--s_threshold", default=0.2, type=float)
    parser.add_argument("--data_queue_size", default=5, type=int)
    parser.add_argument("--data_queue_size", default=5, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    picker = S3MongoSBPicker(**vars(args))

    picker.run()


class S3MongoSBPicker:
    def __init__(
        self,
        s3: str,
        s3_format: str,
        db_uri: str,
        db_container: str,
        stations: str,
        start: datetime.date,
        end: datetime.date,
        components: str,
        model: str,
        weight: str,
        p_threshold: float,
        s_threshold: float,
        data_queue_size: Optional[int],
        pick_queue_size: Optional[int],
        debug: bool = False,
    ):
        if debug:
            logger.setLevel(logging.DEBUG)

        self.model = self.create_model(model, weight, p_threshold, s_threshold)
        self.stations = stations.split(",")

        self.s3 = s3
        self.s3_format = s3_format
        self.db_uri = db_uri
        self.container_db = db_container
        self.start = start
        self.end = end
        self.data_queue_size = data_queue_size
        self.pick_queue_size = pick_queue_size
        self.components = components

        self.metadata = {
            "model": model,
            "weight": weight,
            "p_threshold": p_threshold,
            "s_threshold": s_threshold,
            "components_loaded": components,
            "seisbench_version": seisbench.__version,
            "weight_version": self.model.weights_version,
        }

        self._db = None

    @property
    def db(self) -> pymongo.MongoClient:
        if self._db is None:
            self._db = pymongo.MongoClient(self.db_uri)
            db, container = self.container_db.split(".")
            if "unique_index" not in self._db[db][container].index_information():
                self._db[db][container].create_index(
                    ["trace_id", "phase", "time"], unique=True, name="unique_index"
                )

        return self._db

    @property
    def collection(self) -> pymongo.collection.Collection:
        db, container = self.container_db.split(".")
        return self.db[db][container]

    @staticmethod
    def create_model(
        model: str, weight: str, p_threshold: float, s_threshold: float
    ) -> sbm.WaveformModel:
        model = sbm.__getattribute__(model).from_pretrained(weight)
        model.default_args["P_threshold"] = p_threshold
        model.default_args["S_threshold"] = s_threshold
        return model

    @staticmethod
    def parse_stations(stations: str) -> Iterable:
        pass

    def run(self):
        days = np.arange(self.start, self.end, datetime.timedelta(days=1))

        tasks = asyncio.Queue()
        for station in self.stations:
            for day in days:
                tasks.put((station, day))
        tasks.put(None)

        asyncio.run(self._run_async(tasks))

    async def _run_async(self, tasks: asyncio.Queue):
        data = asyncio.Queue(self.data_queue_size)
        picks = asyncio.Queue(self.pick_queue_size)

        task_load = self._load_data(tasks, data)
        task_pick = self._pick_data(data, picks)
        task_db = self._write_pick_to_db(picks)

        await asyncio.gather(task_load, task_pick, task_db)

    async def _load_data(
        self,
        tasks: asyncio.Queue[tuple[str, datetime.date] | None],
        data: asyncio.Queue[obspy.Stream | None],
    ):
        fs = S3FileSystem(anon=True)
        while True:
            task = await tasks.get()
            if task is None:
                await data.put(None)
                break

            station, date = task
            logger.debug(f"Loading {station} - {date}")

            stream = obspy.Stream()
            for uri in self._generate_waveform_uris(station, date):
                stream += await asyncio.to_thread(self._read_from_s3, fs, uri)

    @staticmethod
    def _read_from_s3(fs, uri) -> obspy.Stream:
        try:
            with fs.open(uri) as f:
                return obspy.read(f)
        except FileNotFoundError:
            return obspy.Stream()

    def _generate_waveform_uris(self, station: str, date: datetime.date) -> list[str]:
        uris = []
        if self.s3_format == "ncedc":
            net, sta, loc, cha = station.split(".")
            year = date.strftime("%Y")
            day = date.strftime("%j")
            for c in self.components:
                f"ncedc-pds/continuous_waveforms/{net}/{year}/{year}.{day}/{sta}.{net}.{cha}{c}.{loc}.D.{year}.{day}"
        else:
            raise NotImplementedError(f"Format '{format}' unknown.")

        return uris

    async def _pick_data(
        self,
        data: asyncio.Queue[obspy.Stream | None],
        picks: asyncio.Queue[sbu.PickList | None],
    ):
        while True:
            stream = await data.get()
            if stream is None:
                await picks.put(None)
                break

            if len(stream) == 0:
                continue

            logger.debug(f"Picking {stream[0].id} - {stream[0].stats.starttime}")

            stream_annotations = await asyncio.to_thread(self.model.classify, stream)
            await picks.put(stream_annotations.picks)

    async def _write_pick_to_db(self, picks: asyncio.Queue[sbu.PickList | None]):
        while True:
            stream_picks = await picks.get()
            if stream_picks is None:
                break

            if len(stream_picks) == 0:
                continue

            logger.debug(
                f"Putting {len(stream_picks)} picks including {stream_picks[0]}"
            )

            await asyncio.to_thread(self._write_single_picklist_to_db, stream_picks)

    def _write_single_picklist_to_db(self, picks: sbu.PickList):
        try:
            self.collection.insert_many(
                [
                    {
                        "trace_id": pick.trace_id,
                        "time": pick.peak_time.datetime,
                        "confidence": pick.peak_value,
                        "phase": pick.phase,
                        **self.metadata,
                    }
                    for pick in picks
                ],
                ordered=False,  # Not ordered to make sure every query is sent
            )
        except DuplicateKeyError:
            logger.warning("Some duplicate entries have been skipped.")


if __name__ == "__main__":
    main()
