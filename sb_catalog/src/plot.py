"""
Plot results
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from .util import SeisBenchCollection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_uri", type=str)
    parser.add_argument("--collection", type=str, default="tutorial")
    args = parser.parse_args()

    plot_events(args.db_uri, args.collection)


def plot_events(db_uri: str, collection: str) -> None:
    db = SeisBenchCollection(db_uri, collection)

    cursor = db["events"].find()
    events = pd.DataFrame(list(cursor))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cb = ax.scatter(events["x"], events["y"], c=events["depth"], vmin=0, s=4)
    ax.set_aspect("equal")
    ax.set_xlabel("East [km]")
    ax.set_ylabel("North [km]")
    cbar = fig.colorbar(cb, label="Depth [km]")
    cbar.ax.invert_yaxis()
    fig.savefig("events.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
