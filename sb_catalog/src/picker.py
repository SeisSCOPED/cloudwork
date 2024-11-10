import argparse
import asyncio
import datetime
import functools
import itertools
import logging
import re
from typing import Any, AsyncIterator, Optional

import numpy as np
import obspy
import pandas as pd
import pyocto
import seisbench
import seisbench.models as sbm
import seisbench.util as sbu
from bson import ObjectId
from s3fs import S3FileSystem
from tqdm import tqdm

from .util import SeisBenchDatabase, network_mapper, s3_path_mapper, parse_year_day

logger = logging.getLogger("sb_picker")

def main() -> None:
    """
    This main function serves as the entry point to all functionality available in the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", type=str, required=True,
        help="Subroutine to execute. See below for available functions."
    )
    parser.add_argument(
        "--db_uri", type=str, required=True, help="URI of the MongoDB cluster."
    )
    parser.add_argument(
        "--database", type=str, required=True, help="MongoDB database name."
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
        help="Format: YYYY.DDD (included).",
    )
    parser.add_argument(
        "--end",
        type=parse_year_day,
        required=False,
        help="Format: YYYY.DDD (not included).",
    )
    parser.add_argument(
        "--extent",
        type=str,
        required=False,
        help="Comma separated: minlat, maxlat, minlon, maxlon",
    )
    parser.add_argument(
        "--components", type=str, default="ZNE12", help="Components to scan."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PhaseNet",
        help="Model type. Must be available in SeisBench.",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default="instance",
        help="Model weights to load through SeisBench from_pretrained.",
    )
    parser.add_argument(
        "--p_threshold", default=0.2, type=float, help="Picking threshold for P waves."
    )
    parser.add_argument(
        "--s_threshold", default=0.2, type=float, help="Picking threshold for S waves."
    )
    parser.add_argument(
        "--data_queue_size",
        default=5,
        type=int,
        help="Buffer size for data preloading.",
    )
    parser.add_argument(
        "--pick_queue_size",
        default=5,
        type=int,
        help="Buffer size for picking results.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enables additional debug output."
    )
    args = parser.parse_args()

    if args.debug:  # Setup debug logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    # Set up data base for results and data source
    db = SeisBenchDatabase(args.db_uri, args.database)
    s3 = S3DataSource(
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

    # Set up main class handling the commands
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


class S3DataSource:
    """
    This class provides functionality to load waveform data from an S3 bucket.
    """

    def __init__(
        self,
        start: Optional[datetime.date] = None,
        end: Optional[datetime.date] = None,
        stations: Optional[str] = None,
        components: str = "ZNE12",
        channels: str = "HH?,BH?,EH?,HN?,BN?,EN?,HL?,BL?,EL?",
    ):
        self.start = start
        self.end = end
        self.components = components
        self.s3s = list(set(network_mapper.values()))
        if stations is None:
            self.stations = []
        else:
            self.stations = stations.split(",")
        self.channels = channels.split(",")

    async def load_waveforms(self) -> AsyncIterator[obspy.Stream]:
        """
        Load the waveforms. This function is async to allow loading data in parallel with processing.
        The function releases the GIL when reading from the S3 bucket.
        The iterator returns data by station and within each station day by day.
        Data from all channels of a station is returned simultaneously.
        This matches the typical access pattern required for single-station phase pickers.
        """
        days = np.arange(self.start, self.end, datetime.timedelta(days=1))
        fs = S3FileSystem(anon=True)

        for station in self.stations:
            for day in days:
                day = day.astype(datetime.datetime)
                logger.debug(f"Loading {station} - {day.strftime('%Y.%j')}")
                stream = obspy.Stream()
                for channel in self.channels:
                    for uri in self._generate_waveform_uris(station, channel[:2], day):
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
        """
        List all stations available in the S3 bucket by scanning the StationXML files.
        Returns station list as a dataframe.
        """
        fs = S3FileSystem(anon=True)

        station_uris = []
        for s3 in self.s3s:
            logger.debug(f"Listing StationXML URIs from {s3}.")
            networks = fs.ls(f"{s3}/FDSNstationXML/")
            for net in networks:
                # SCEDC also holds "unauthoritative-XML" for other stations
                if len(net.split("/")[-1]) == 2:
                    station_uris += fs.ls(net)[1:]

        logger.debug("Reading and parsing all station inventories. This may take some time...")
        stations = []
        for uri in tqdm(station_uris, total=len(station_uris)):
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

        def unify_channel_names(x):
            """
            Combines channel names into a string
            """
            channels = itertools.chain.from_iterable(
                channels.split(",") for channels in x["channels"]
            )
            channels = ",".join(sorted(list(set(channels))))
            out = x.iloc[0].copy()
            out["channels"] = channels
            return out

        stations = (
            stations.groupby("id")
            .apply(unify_channel_names, include_groups=False)
            .reset_index()
        )

        return stations

    def _check_channel(self, channel: str) -> bool:
        """
        Check whether a channel matches the channel patterns defined in `self.channels`
        """
        for pattern in self.channels:
            pattern = pattern.replace("?", ".?").replace("*", ".*")
            if re.fullmatch(pattern, channel):
                return True
        return False

    @staticmethod
    def _read_waveform_from_s3(fs, uri) -> obspy.Stream:
        """
        Failure tolerant method for reading data from S3. If an error occurs, an empty stream is returned.
        """
        try:
            with fs.open(uri) as f:
                return obspy.read(f)
        except FileNotFoundError:  # File does not exist
            return obspy.Stream()
        except ValueError:  # Raised for certain types of corrupt files
            return obspy.Stream()

    def _generate_waveform_uris(
        self, station: str, cha: str, date: datetime.date
    ) -> list[str]:
        """
        Generates a list of S3 uris for the requested data
        """
        uris = []
        net, sta, loc = station.split(".")
        year = date.strftime("%Y")
        day = date.strftime("%j")
        for c in self.components:
            # go through all possible components...
            uris.append(s3_path_mapper(net, sta, loc, cha, year, day, c))

        return uris


class S3MongoSBBridge:
    """
    This bridge connects an S3DataSource, a MongoDB database (represented by the SeisBenchDatabase) and
    the processing for picking and association (implemented directly in the class).
    Additional functionality is provided for submitting jobs to AWS Batch, however, these functions are also
    available separately in submit.py.
    """

    def __init__(
        self,
        s3: S3DataSource,
        db: SeisBenchDatabase,
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

        self.station_group_size = 8
        self.day_group_size = 2

        self._run_id = None

    @property
    def run_id(self) -> ObjectId:
        """
        A unique run_id that is saved in the database along with the configuration for reproducibility.
        """
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
        return self.db.database["sb_runs"].insert_one(kwargs).inserted_id

    @staticmethod
    def create_model(
        model: str, weight: str, p_threshold: float, s_threshold: float
    ) -> sbm.WaveformModel:
        """
        Loads a SeisBench model
        """
        model = sbm.__getattribute__(model).from_pretrained(weight)
        model.default_args["P_threshold"] = p_threshold
        model.default_args["S_threshold"] = s_threshold
        return model

    def run_station_listing(self):
        """
        Lists all available stations and writes them to the database.
        """
        stations = self.s3.get_available_stations()
        self._write_stations_to_db(stations)

    def _write_stations_to_db(self, stations):
        logger.debug("Writing station information to MongoDB")
        self.db.insert_many_ignore_duplicates("stations", stations.to_dict("records"))

    def run_association(self, t0: datetime.datetime, t1: datetime.datetime):
        """
        Runs the phase association for the provided time range and the extent defined in self.extent.
        """
        t0 = self._date_to_datetime(t0)
        t1 = self._date_to_datetime(t1)
        stations = self.db.get_stations(self.extent)
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
        """
        Helper function to homogenize time formats
        """
        if isinstance(t, datetime.date):
            return datetime.datetime.combine(t, datetime.datetime.min.time())
        return t

    def _write_events_to_db(
        self, events: pd.DataFrame, assignments: pd.DataFrame, picks: pd.DataFrame
    ) -> None:
        """
        Writes events and the associated picks into the MongoDB. Ensures that the pick and event ids are consistent
        with the ones used in the database.
        """
        # Put events and get mongodb ids, replace event and pick ids with their mongodb counterparts,
        # write assignments to database
        event_result = self.db.insert_many_ignore_duplicates(
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

        self.db.database["assignments"].insert_many(merged.to_dict("records"))

    def _load_picks(
        self, station_ids: list[str], t0: datetime.datetime, t1: datetime.datetime
    ) -> pd.DataFrame:
        """
        Loads picks for a list of stations during a given time range from the database.
        The database has already been configured with indices that speed up this query.
        """
        cursor = self.db.database["picks"].find(
            {
                "time": {"$gt": t0, "$lt": t1},
                "trace_id": {"$in": station_ids},
            }
        )

        return pd.DataFrame(cursor)

    def run_picking(self) -> None:
        """
        Perform the picking
        """
        asyncio.run(self._run_picking_async())

    async def _run_picking_async(self) -> None:
        """
        An async implementation of the data loading, picking, and output routine.
        All three tasks are started in parallel with buffer queues in between.
        This means that the next input data is loaded while the current one is picked.
        Similarly, the outputs are written to MongoDB while the next data is already being processed.
        To guarantee this, all underlying functions have been designed to release the GIL.
        """
        data = asyncio.Queue(self.data_queue_size)
        picks = asyncio.Queue(self.pick_queue_size)

        task_load = self._load_data(data)
        task_pick = self._pick_data(data, picks)
        task_db = self._write_picks_to_db(picks)

        await asyncio.gather(task_load, task_pick, task_db)

    async def _load_data(
        self,
        data: asyncio.Queue[obspy.Stream | None],
    ) -> None:
        """
        An async function getting data from the S3 sources and putting it into a queue.
        """
        async for stream in self.s3.load_waveforms():
            await data.put(stream)

        await data.put(None)

    async def _pick_data(
        self,
        data: asyncio.Queue[obspy.Stream | None],
        picks: asyncio.Queue[sbu.PickList | None],
    ) -> None:
        """
        An async function taking data from a queue, picking it and returning the results to an output queue.
        """
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

    async def _write_picks_to_db(
        self, picks: asyncio.Queue[sbu.PickList | None]
    ) -> None:
        """
        An async function reading picks from a queue and putting them into the MongoDB.
        """
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
        """
        Converts picks into records that can be submitted to MongoDB and writes them.
        """
        self.db.insert_many_ignore_duplicates(
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

    def get_pick_jobs(self) -> None:
        """
        Lists the available stations in an area and prints them
        :return:
        """
        stations = self.db.get_stations(self.extent)
        logger.debug(f"Found {len(stations)} jobs")
        print(",".join(stations["id"]))


if __name__ == "__main__":
    main()
