"""
Take an existing dataset and delete some demonstration data.

NOTE: you should run h5repack afterwards to free up space!
"""
import argparse
import h5py
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to dataset to modify",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="number of demos to keep",
    )
    args = parser.parse_args()


    f = h5py.File(args.dataset, "a")

    # get last N demos
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    demos = demos[-args.n:]

    # delete all but last N demos
    total_samples = 0
    for demo in f["data"]:
        if demo in demos:
            total_samples += f["data/{}/actions".format(demo)].shape[0]
        else:
            del f["data/{}".format(demo)]
    f["data"].attrs["total"] = total_samples
    f.close()
