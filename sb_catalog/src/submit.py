import argparse
import datetime
import logging

import boto3
import numpy as np

from .parameters import JOB_DEFINITION_ASSOCIATION, JOB_DEFINITION_PICKING, JOB_QUEUE
from .util import SeisBenchDatabase

logger = logging.getLogger("sb_picker")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class SubmitHelper:
    """
    A helper class to submit picking and association jobs.

    Make sure the job queue and job names are set in the parameters file

    :param start: Start date
    :param end: End date
    :param extent: Study area (minlat, maxlat, minlon, maxlon)
    :param station_group_size: Number of stations to process in a single picking job
    :param day_group_size: Number of days to process in a single picking/association job
    """

    def __init__(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        extent: tuple[float, float, float, float],
        db: SeisBenchDatabase,
        station_group_size: int = 8,
        day_group_size: int = 4,
    ):
        self.start = start
        self.end = end
        self.extent = extent
        self.db = db
        self.station_group_size = station_group_size
        self.day_group_size = day_group_size

    def submit_jobs(self) -> None:
        stations = self.db.get_stations(self.extent)
        days = np.arange(self.start, self.end, datetime.timedelta(days=1))
        logger.debug(f"Starting jobs for {len(stations)} stations and {len(days)} days")

        client = boto3.client("batch")

        shared_parameters = {
            "db_uri": self.db.db_uri,
            "database": self.db.database,
        }

        i = 0
        while i < len(days) - 1:
            day0 = days[i].astype(datetime.datetime).strftime("%Y.%j")
            day1 = (
                days[min(i + self.day_group_size, len(days) - 1)]
                .astype(datetime.datetime)
                .strftime("%Y.%j")
            )

            pick_jobs = []
            j = 0
            while j < len(stations) - 1:
                sub_stations = ",".join(
                    stations["id"].iloc[j : j + self.station_group_size]
                )
                parameters = {"start": day0, "end": day1, "stations": sub_stations}

                logger.debug(f"Submitting pick job with: {parameters}")
                pick_jobs.append(
                    client.submit_job(
                        jobName=f"munchmeyer_picking_{i}_{j}",
                        jobQueue=JOB_QUEUE,
                        jobDefinition=JOB_DEFINITION_PICKING,
                        parameters={**parameters, **shared_parameters},
                    )
                )

                j += self.station_group_size

            extent = ",".join([str(x) for x in self.extent])
            parameters = {"start": day0, "end": day1, "extent": extent}
            logger.debug(f"Submitting association job with: {parameters}")
            dependencies = [
                {"jobId": job["jobId"], "type": "SEQUENTIAL"} for job in pick_jobs
            ]
            pick_jobs.append(
                client.submit_job(
                    jobName=f"munchmeyer_association_{i}",
                    jobQueue=JOB_QUEUE,
                    jobDefinition=JOB_DEFINITION_ASSOCIATION,
                    dependsOn=dependencies,
                    parameters={**parameters, **shared_parameters},
                )
            )

            i += self.day_group_size


def parse_year_day(x: str) -> datetime.date:
    return datetime.datetime.strptime(x, "%Y.%j").date()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "start",
        type=parse_year_day,
        help="Format: YYYY.DDD (included)",
    )
    parser.add_argument(
        "end",
        type=parse_year_day,
        help="Format: YYYY.DDD (not included)",
    )
    parser.add_argument(
        "extent",
        type=str,
        help="Comma separated: minlat, maxlat, minlon, maxlon",
    )
    parser.add_argument("db_uri", type=str)
    parser.add_argument("--database", type=str, default="tutorial")

    args = parser.parse_args()

    extent = tuple([float(x) for x in args.extent.split(",")])
    assert len(extent) == 4, "Extent needs to be exactly 4 coordinates"

    db = SeisBenchDatabase(args.db_uri, args.database)
    helper = SubmitHelper(start=args.start, end=args.end, extent=extent, db=db)
    helper.submit_jobs()


if __name__ == "__main__":
    main()
