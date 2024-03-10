import argparse
import asyncio
import datetime
import functools
import itertools
import logging
import re
from typing import Any, AsyncIterator, Iterable, Mapping, Optional

import numpy as np
import obspy
import pandas as pd
import pymongo
import pyocto
import seisbench
import seisbench.models as sbm
import seisbench.util as sbu
from bson import ObjectId
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, DuplicateKeyError
from pymongo.results import InsertManyResult
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
    parser.add_argument(
        "--extent",
        type=str,
        required=False,
        help="Comma separated: minlat, maxlat, minlon, maxlon",
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
    if args.extent is None:
        extent = None
    else:
        extent = tuple([float(x) for x in args.extent.split(",")])
        assert len(extent) == 4, "Extent needs to be exactly 4 coordinates"

    picker = S3MongoSBBridge(
        s3=s3,
        db=db,
        model=args.model,
        weight=args.weight,
        p_threshold=args.p_threshold,
        s_threshold=args.s_threshold,
        data_queue_size=args.data_queue_size,
        pick_queue_size=args.pick_queue_size,
        extent=extent,
    )

    if args.command == "pick":
        picker.run_picking()
    elif args.command == "station_list":
        picker.run_station_listing()
    elif args.command == "associate":
        picker.run_association(args.start, args.end)
    elif args.command == "pick_jobs":
        picker.get_pick_jobs()
    else:
        raise ValueError(f"Unknown command '{args.command}'")


class SeisBenchCollection(pymongo.MongoClient):
    def __init__(self, db_uri: str, collection: str, **kwargs: Any) -> None:
        super().__init__(db_uri, **kwargs)

        self.db_uri = db_uri
        self._collection = super().__getitem__(collection)

        self.dbs = {"picks", "stations", "sb_runs", "events", "assignments"}
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
        channels: str = "HH?,BH?,EH?,HN?,BN?,EN?,HL?,BL?,EL?",
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

    async def load_waveforms(self) -> AsyncIterator[obspy.Stream]:
        days = np.arange(self.start, self.end, datetime.timedelta(days=1))
        fs = S3FileSystem(anon=True)

        for station in self.stations:
            for day in days:
                logger.debug(f"Loading {station} - {day}")
                stream = obspy.Stream()
                for channel in self.channels:
                    for uri in self._generate_waveform_uris(
                        station,
                        channel[
                            :2
                        ],  # Truncate the ? from the end of the channels string
                        day.astype(datetime.datetime),
                    ):
                        stream += await asyncio.to_thread(
                            self._read_waveform_from_s3, fs, uri
                        )
                    if len(stream) > 0:
                        break

                if len(stream) > 0:
                    yield stream
                else:
                    logger.debug(f"Empty stream {station} - {day}")

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

    def _generate_waveform_uris(
        self, station: str, channel: str, date: datetime.date
    ) -> list[str]:
        uris = []
        if self.s3_format == "ncedc":
            net, sta, loc = station.split(".")
            year = date.strftime("%Y")
            day = date.strftime("%j")
            for c in self.components:
                uris.append(
                    f"{self.s3}/continuous_waveforms/{net}/{year}/{year}.{day}/{sta}.{net}.{channel}{c}.{loc}.D.{year}.{day}"
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
        extent: Optional[tuple[float, float, float, float]] = None,
    ):
        self.extent = extent
        if model is not None:
            self.model = self.create_model(model, weight, p_threshold, s_threshold)
        else:
            self.model = None
        self.model_name = model
        self.weight = weight
        self.p_threshold = p_threshold
        self.s_threshold = s_threshold

        self.s3 = s3
        self.db = db

        self.data_queue_size = data_queue_size
        self.pick_queue_size = pick_queue_size

        self._run_id = None

    @property
    def run_id(self) -> ObjectId:
        if self._run_id is None:
            self._run_id = self._put_run_data(
                model=self.model_name,
                weight=self.weight,
                p_threshold=self.p_threshold,
                s_threshold=self.s_threshold,
                components_loaded=self.s3.components,
                seisbench_version=seisbench.__version__,
                weight_version=self.model.weights_version,
            )
        return self._run_id

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
        self._write_stations_to_db(stations)

    def _write_stations_to_db(self, stations):
        logger.debug("Writing station information to DB")
        self._insert_many_ignore_duplicates("stations", stations.to_dict("records"))

    def run_association(self, t0: datetime.datetime, t1: datetime.datetime):
        t0 = self._date_to_datetime(t0)
        t1 = self._date_to_datetime(t1)
        stations = self._get_stations()
        logger.debug(
            f"Associating {len(stations)} stations: " + ",".join(stations["id"].values)
        )

        picks = self._load_picks(list(stations["id"].values), t0, t1)
        picks.rename(columns={"trace_id": "station"}, inplace=True)
        picks["time"] = picks["time"].apply(lambda x: x.timestamp())
        logger.debug(f"Associating {len(picks)} picks")

        if len(picks) == 0:
            logger.warning("Found no picks, exiting")
            return

        minlat, maxlat, minlon, maxlon = self.extent
        # TODO: PyOcto configuration
        velocity_model = pyocto.VelocityModel0D(
            p_velocity=6.0,
            s_velocity=6.0 / 1.75,
            tolerance=1.5,
            association_cutoff_distance=150,
        )
        associator = pyocto.OctoAssociator.from_area(
            (minlat, maxlat),
            (minlon, maxlon),
            (0, 50),
            velocity_model,
            time_before=150,
        )
        stations = associator.transform_stations(stations)

        events, assignments = associator.associate(picks, stations)
        logger.debug(
            f"Found {len(events)} events with {len(assignments)} total picks (of {len(picks)} input picks)"
        )

        utc_from_timestamp = functools.partial(
            datetime.datetime.fromtimestamp, tz=datetime.timezone.utc
        )
        if len(events) > 0:
            events = associator.transform_events(events)
            events["time"] = events["time"].apply(utc_from_timestamp)

        self._write_events_to_db(events, assignments, picks)

    @staticmethod
    def _date_to_datetime(t: datetime.date | datetime.datetime) -> datetime.datetime:
        if isinstance(t, datetime.date):
            return datetime.datetime.combine(t, datetime.datetime.min.time())
        return t

    def _write_events_to_db(
        self, events: pd.DataFrame, assignments: pd.DataFrame, picks: pd.DataFrame
    ) -> None:
        # Put events and get mongodb ids, replace event and pick ids with their mongodb counterparts,
        # write assignments to database
        event_result = self._insert_many_ignore_duplicates(
            "events", events.to_dict("records")
        )

        event_key = pd.DataFrame(
            {
                "event_id": event_result.inserted_ids,
                "event_idx": events["idx"].values,
            }
        )
        pick_key = pd.DataFrame(
            {
                "pick_id": picks["_id"],
                "pick_idx": np.arange(len(picks)),
            }
        )

        merged = pd.merge(event_key, assignments, on="event_idx")
        merged = pd.merge(merged, pick_key, on="pick_idx")

        merged = merged[["event_id", "pick_id"]]

        self.db["assignments"].insert_many(merged.to_dict("records"))

    def _get_stations(self) -> pd.DataFrame:
        minlat, maxlat, minlon, maxlon = self.extent

        cursor = self.db["stations"].find(
            {
                "latitude": {"$gt": minlat, "$lt": maxlat},
                "longitude": {"$gt": minlon, "$lt": maxlon},
            }
        )

        return pd.DataFrame(list(cursor))

    def _load_picks(
        self, station_ids: list[str], t0: datetime.datetime, t1: datetime.datetime
    ) -> pd.DataFrame:
        cursor = self.db["picks"].find(
            {
                "time": {"$gt": t0, "$lt": t1},
                "trace_id": {"$in": station_ids},
            }
        )

        return pd.DataFrame(cursor)

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
        async for stream in self.s3.load_waveforms():
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

    def _write_single_picklist_to_db(self, picks: sbu.PickList) -> None:
        self._insert_many_ignore_duplicates(
            "picks",
            [
                {
                    "trace_id": pick.trace_id,
                    "time": pick.peak_time.datetime,
                    "confidence": float(pick.peak_value),
                    "phase": pick.phase,
                    "run_id": self.run_id,
                }
                for pick in picks
            ],
        )

    def _insert_many_ignore_duplicates(
        self, key: str, entries: list[dict[str, Any]]
    ) -> InsertManyResult:
        try:
            return self.db[key].insert_many(
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

    def get_pick_jobs(self) -> None:
        stations = self._get_stations()
        logger.debug(f"Found {len(stations)} jobs")
        print(",".join(stations["id"]))


if __name__ == "__main__":
    main()
