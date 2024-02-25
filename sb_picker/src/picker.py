import argparse
import asyncio
import datetime
import itertools
import logging
import re
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import obspy
import pandas as pd
import pymongo
import seisbench
import seisbench.models as sbm
import seisbench.util as sbu
from bson import ObjectId
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, DuplicateKeyError
from s3fs import S3FileSystem
from tqdm import tqdm

logger = logging.getLogger("sb_picker")


def parse_year_day(x: str) -> datetime.date:
    return datetime.datetime.strptime(x, "%Y.%j").date()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("--s3", type=str, required=True)
    parser.add_argument("--s3_format", type=str, default="ncedc")
    parser.add_argument("--db_uri", type=str, required=True)
    parser.add_argument(
        "--collection", type=str, required=True, help="The collection for MongoDB"
    )
    parser.add_argument(
        "--stations",
        type=str,
        required=False,
        help="Stations (comma separated) in format NET.STA.LOC.CHA without component.",
    )
    parser.add_argument(
        "--start",
        type=parse_year_day,
        required=False,
        help="Format: YYYY.DDD (included)",
    )
    parser.add_argument(
        "--end",
        type=parse_year_day,
        required=False,
        help="Format: YYYY.DDD (not included)",
    )
    parser.add_argument("--components", type=str, default="ZNE12")
    parser.add_argument("--model", type=str, default="PhaseNet")
    parser.add_argument("--weight", type=str, default="instance")
    parser.add_argument("--p_threshold", default=0.2, type=float)
    parser.add_argument("--s_threshold", default=0.2, type=float)
    parser.add_argument("--data_queue_size", default=5, type=int)
    parser.add_argument("--pick_queue_size", default=5, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    db = SeisBenchCollection(args.db_uri, args.collection)
    s3 = S3DataSource(
        s3=args.s3,
        s3_format=args.s3_format,
        stations=args.stations,
        start=args.start,
        end=args.end,
        components=args.components,
    )
    picker = S3MongoSBBridge(
        s3=s3,
        db=db,
        model=args.model,
        weight=args.weight,
        p_threshold=args.p_threshold,
        s_threshold=args.s_threshold,
        data_queue_size=args.data_queue_size,
        pick_queue_size=args.pick_queue_size,
    )

    if args.command == "pick":
        picker.run_picking()
    elif args.command == "station_list":
        picker.run_station_listing()
    else:
        raise ValueError(f"Unknown command '{args.command}'")


class SeisBenchCollection(pymongo.MongoClient):
    def __init__(self, db_uri: str, collection: str, **kwargs: Any) -> None:
        super().__init__(db_uri, **kwargs)

        self.db_uri = db_uri
        self._collection = super().__getitem__(collection)

        self.dbs = {"picks", "stations", "sb_runs"}
        self._setup()

    def _setup(self):
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


class S3DataSource:
    def __init__(
        self,
        s3: str,
        s3_format: str,
        start: Optional[datetime.date] = None,
        end: Optional[datetime.date] = None,
        stations: Optional[str] = None,
        components: str = "ZNE12",
        channels: str = "HH?,BH?,EH?,HN?,BN?,EN?,HL?,BL?,EL?",  # Only for station parsing
    ):
        self.s3 = s3
        self.s3_format = s3_format
        self.start = start
        self.end = end
        self.components = components
        if stations is None:
            self.stations = []
        else:
            self.stations = stations.split(",")
        self.channels = channels.split(",")

    def load_waveforms(self) -> obspy.Stream:
        days = np.arange(self.start, self.end, datetime.timedelta(days=1))
        fs = S3FileSystem(anon=True)

        for station in self.stations:
            for day in days:
                logger.debug(f"Loading {station} - {day}")
                stream = obspy.Stream()
                for uri in self._generate_waveform_uris(
                    station, day.astype(datetime.datetime)
                ):
                    stream += self._read_waveform_from_s3(fs, uri)
                yield stream

    def get_available_stations(self) -> pd.DataFrame:
        fs = S3FileSystem(anon=True)
        logger.debug("Listing station URIs")
        networks = fs.ls(f"{self.s3}/FDSNstationXML/")[1:]
        station_uris = []
        for net in networks:
            station_uris += fs.ls(net)[1:]

        logger.debug("Parsing station inventories")
        stations = []
        for uri in tqdm(station_uris):
            with fs.open(uri) as f:
                inv = obspy.read_inventory(f)

            for net in inv:
                for sta in net:
                    locs = {cha.location_code for cha in sta}
                    for loc in locs:
                        channels = ",".join(
                            sorted(
                                {
                                    cha.code
                                    for cha in sta.select(location=loc)
                                    if self._check_channel(cha.code)
                                }
                            )
                        )
                        if channels == "":
                            continue

                        stations.append(
                            {
                                "network_code": net.code,
                                "station_code": sta.code,
                                "location_code": loc,
                                "channels": channels,
                                "id": f"{net.code}.{sta.code}.{loc}",
                                "latitude": sta.latitude,
                                "longitude": sta.longitude,
                                "elevation": sta.elevation,
                            }
                        )

        stations = pd.DataFrame(stations)

        def unify(x):
            channels = itertools.chain.from_iterable(
                channels.split(",") for channels in x["channels"]
            )
            channels = ",".join(sorted(list(set(channels))))
            out = x.iloc[0].copy()
            out["channels"] = channels
            return out

        stations = (
            stations.groupby("id").apply(unify, include_groups=False).reset_index()
        )

        return stations

    def _check_channel(self, channel: str) -> bool:
        for pattern in self.channels:
            pattern = pattern.replace("?", ".?").replace("*", ".*")
            if re.fullmatch(pattern, channel):
                return True
        return False

    @staticmethod
    def _read_waveform_from_s3(fs, uri) -> obspy.Stream:
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
                uris.append(
                    f"{self.s3}/continuous_waveforms/{net}/{year}/{year}.{day}/{sta}.{net}.{cha}{c}.{loc}.D.{year}.{day}"
                )
        else:
            raise NotImplementedError(f"Format '{format}' unknown.")

        return uris


class S3MongoSBBridge:
    def __init__(
        self,
        s3: S3DataSource,
        db: SeisBenchCollection,
        model: Optional[str] = None,
        weight: Optional[str] = None,
        p_threshold: Optional[float] = None,
        s_threshold: Optional[float] = None,
        data_queue_size: Optional[int] = None,
        pick_queue_size: Optional[int] = None,
    ):
        if model is not None:
            self.model = self.create_model(model, weight, p_threshold, s_threshold)

            self.s3 = s3
            self.db = db

            self.data_queue_size = data_queue_size
            self.pick_queue_size = pick_queue_size

            self._run_id = self._put_run_data(
                model=model,
                weight=weight,
                p_threshold=p_threshold,
                s_threshold=s_threshold,
                components_loaded=s3.components,
                seisbench_version=seisbench.__version__,
                weight_version=self.model.weights_version,
            )

    def _put_run_data(self, **kwargs: Any) -> ObjectId:
        kwargs["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
        return self.db["sb_runs"].insert_one(kwargs).inserted_id

    @staticmethod
    def create_model(
        model: str, weight: str, p_threshold: float, s_threshold: float
    ) -> sbm.WaveformModel:
        model = sbm.__getattribute__(model).from_pretrained(weight)
        model.default_args["P_threshold"] = p_threshold
        model.default_args["S_threshold"] = s_threshold
        return model

    def run_station_listing(self):
        stations = self.s3.get_available_stations()
        logger.debug("Writing station information to DB")
        try:
            self.db["stations"].insert_many(
                stations.to_dict("records"),
                ordered=False,  # Not ordered to make sure every query is sent
            )
        except DuplicateKeyError:
            logger.warning("Some duplicate entries have been skipped.")
        except BulkWriteError as e:
            if all(x["code"] == 11000 for x in e.details["writeErrors"]):
                logger.warning("Some duplicate entries have been skipped.")
            else:
                raise e

    def run_picking(self):
        asyncio.run(self._run_picking_async())

    async def _run_picking_async(self):
        data = asyncio.Queue(self.data_queue_size)
        picks = asyncio.Queue(self.pick_queue_size)

        task_load = self._load_data(data)
        task_pick = self._pick_data(data, picks)
        task_db = self._write_picks_to_db(picks)

        await asyncio.gather(task_load, task_pick, task_db)

    async def _load_data(
        self,
        data: asyncio.Queue[obspy.Stream | None],
    ):
        for stream in self.s3.load_waveforms():
            await data.put(stream)

        await data.put(None)

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

    async def _write_picks_to_db(self, picks: asyncio.Queue[sbu.PickList | None]):
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
            self.db["picks"].insert_many(
                [
                    {
                        "trace_id": pick.trace_id,
                        "time": pick.peak_time.datetime,
                        "confidence": float(pick.peak_value),
                        "phase": pick.phase,
                        "run_id": self._run_id,
                    }
                    for pick in picks
                ],
                ordered=False,  # Not ordered to make sure every query is sent
            )
        except DuplicateKeyError:
            logger.warning("Some duplicate entries have been skipped.")
        except BulkWriteError as e:
            if all(x["code"] == 11000 for x in e.details["writeErrors"]):
                logger.warning("Some duplicate entries have been skipped.")
            else:
                raise e


if __name__ == "__main__":
    main()
