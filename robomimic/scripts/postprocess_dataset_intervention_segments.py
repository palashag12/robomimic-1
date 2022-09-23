"""
Script to postprocess a dataset by splitting each trajectory up into new trajectories 
that only consists of continuous intervention segments. This is useful for taking 
a TAMP-gated dataset and splitting it up so that the policy is only trained on the 
human portions.
"""
import os
import h5py
import argparse
import numpy as np


def get_intervention_segments(interventions):
    """
    Splits interventions list into a list of start and end indices (windows) of continuous intervention segments.
    """
    interventions = interventions.reshape(-1).astype(int)
    # pad before and after to make it easy to count starting and ending intervention segments
    expanded_ints = [False] + interventions.astype(bool).tolist() + [False]
    start_inds = []
    end_inds = []
    for i in range(1, len(expanded_ints)):
        if expanded_ints[i] and (not expanded_ints[i - 1]):
            # low to high edge means start of new window
            start_inds.append(i - 1) # record index in original array which is one less (since we added an element to the beg)
        elif (not expanded_ints[i]) and expanded_ints[i - 1]:
            # high to low edge means end of previous window
            end_inds.append(i - 1) # record index in original array which is one less (since we added an element to the beg)

    # run some sanity checks
    assert len(start_inds) == len(end_inds), "missing window edge"
    assert np.all([np.sum(interventions[s : e]) == (e - s) for s, e in zip(start_inds, end_inds)]), "window computation covers non-interventions"
    assert sum([np.sum(interventions[s : e]) for s, e in zip(start_inds, end_inds)]) == np.sum(interventions), "window computation does not cover all interventions"
    return list(zip(start_inds, end_inds))


def write_intervention_segments_as_trajectories(
    src_ep_grp,
    dst_grp,
    start_ep_write_ind,
):
    """
    Helper function to extract intervention segments from a source demonstration and use their indices to
    write the corresponding subset of each trajectory to a new trajectory.

    Returns:
        end_ep_write_ind (int): updated episode index after writing trajectories to destination file
        num_traj (int): number of trajectories written to destination file
        total_samples (int): total number of samples written to destination file
    """

    # get segments
    interventions = src_ep_grp["interventions"][()].reshape(-1).astype(int)
    segments = get_intervention_segments(interventions)

    ep_write_ind = start_ep_write_ind
    total_samples = 0
    num_traj = len(segments)
    for seg_start_ind, seg_end_ind in segments:
        dst_grp_name = "demo_{}".format(ep_write_ind)
        dst_ep_grp = dst_grp.create_group(dst_grp_name)

        # copy over subsequence from source trajectory into destination trajectory
        keys_to_try_and_copy = ["states", "obs", "next_obs", "rewards", "dones"]
        for k in keys_to_try_and_copy:
            if k in src_ep_grp:
                if isinstance(src_ep_grp[k], h5py.Group):
                    for k2 in src_ep_grp[k]:
                        assert isinstance(src_ep_grp[k][k2], h5py.Dataset)
                        dst_ep_grp.create_dataset("{}/{}".format(k, k2), data=np.array(src_ep_grp[k][k2][seg_start_ind : seg_end_ind]))
                else:
                    assert isinstance(src_ep_grp[k], h5py.Dataset)
                    dst_ep_grp.create_dataset("{}".format(k), data=np.array(src_ep_grp[k][seg_start_ind : seg_end_ind]))

        # manually copy actions since they might need truncation (for OSC actions from padded actions for TAMP joint action space)
        actions = np.array(src_ep_grp["actions"][seg_start_ind : seg_end_ind])
        if actions.shape[-1] != 7:
            actions = actions[..., :7]
        dst_ep_grp.create_dataset("actions", data=actions)

        # copy attributes too
        for k in src_ep_grp.attrs:
            dst_ep_grp.attrs[k] = src_ep_grp.attrs[k]
        dst_ep_grp.attrs["num_samples"] = (seg_end_ind - seg_start_ind)

        # update variables for next iter
        ep_write_ind += 1
        total_samples += dst_ep_grp.attrs["num_samples"]
        print("  wrote trajectory to dst grp {} with num samples {}".format(dst_grp_name, dst_ep_grp.attrs["num_samples"]))

    return ep_write_ind, num_traj, total_samples


def postprocess_dataset_intervention_segments(args):
    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("\ninput file: {}".format(args.dataset))
    print("output file: {}\n".format(output_path))

    ep_write_ind = 0
    num_traj = 0
    total_samples = 0
    for ind in range(len(demos)):
        ep = demos[ind]
        print("src grp: {}".format(ep))
        ep_write_ind, ep_num_traj, ep_total_samples = write_intervention_segments_as_trajectories(
            src_ep_grp=f["data/{}".format(ep)],
            dst_grp=data_grp,
            start_ep_write_ind=ep_write_ind,
        )
        num_traj += ep_num_traj
        total_samples += ep_total_samples

    # TODO: update filter keys based on which source demos created which target demos
    # if "mask" in f:
    #     f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = f["data"].attrs["env_args"] # environment info
    print("\nWrote {} trajectories from src with {} trajectories to {}".format(num_traj, len(demos), output_path))

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    args = parser.parse_args()
    postprocess_dataset_intervention_segments(args)
