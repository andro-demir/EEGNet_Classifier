#!/usr/bin/env python
import numpy as np
from scipy.signal import resample
import pandas as pd
import glob
from scipy.io import loadmat
import pickle
import argparse
import concurrent.futures  # for parallel processing
from tqdm import tqdm

num_eeg_channels = 14
num_conds = 18
num_samples = 400
num_subjects = 11
fs_orig = 1200
fs_new = 300


def extract_data(file):
    '''
    param: file(str), is .mat data file of 1 subject inside the Dataset dir.
    return: data_all_cond(np.array, #trials * #channels * #samples), is the
            preprocessed data of 1 subject
    return: movement_labels(np.array, #trials), is the movement type at each
            trial. rub: 0, and tap: 1
    return: surface_labels(np.array, #trials), is the surface touched at each
            trial. flat: 0, medium-rough: 1, and rough: 2
    return: speed_labels(np.array, #trials), is the speed of touch at each trial
            fast: 0, medium: 1, slow: 2
    '''
    data_all_cond = None
    movement_labels = []
    surface_labels = []
    speed_labels = []
    for cond in tqdm(range(num_conds), ascii=True):
        num_trials = file['EEGSeg_Ch'][0, 0][0, cond].shape[0]
        data_per_cond = np.zeros((num_trials, num_eeg_channels, num_samples))
        # if a trial has long enough samples
        if file['EEGSeg_Ch'][0, 0][0, cond].shape[1] >= num_samples:
            for trial in range(num_trials):  # iterate for each trial
                # get movement label
                movement_labels.append(0) if cond < 9 else movement_labels.append(1)
                # get speed label
                if cond in [0, 1, 2, 9, 10, 11]:
                    speed_labels.append(0)
                elif cond in [3, 4, 5, 12, 13, 14]:
                    speed_labels.append(1)
                else:
                    speed_labels.append(2)
                # get condition label
                if cond % 3 == 0:
                    surface_labels.append(0)
                elif cond % 3 == 1:
                    surface_labels.append(1)
                else:
                    surface_labels.append(2)
                # get trial data
                data = pd.DataFrame(columns=['1', '2', '3', '4', '5', '6', '7',
                                             '8', '9', '10', '11', '12', '13', '14'])
                for ch in range(num_eeg_channels):
                    data.iloc[:, ch] = file['EEGSeg_Ch'][0, ch][0, cond][trial, :]

                data.iloc[:, :num_eeg_channels] -= data.iloc[:, :num_eeg_channels].mean()   # DC offset removal
                data.iloc[:, :num_eeg_channels] *= 1e6  # convert volts to microvolts for EEG channels
                data = data.iloc[:num_samples, :]  # use all channels up to num_samples number of samples
                data_per_cond[trial] = data.to_numpy().T[:num_eeg_channels, :]
            # concatenate all trials
            if data_all_cond is None:
                data_all_cond = data_per_cond
            else:
                data_all_cond = np.concatenate((data_all_cond, data_per_cond), axis=0)
    movement_labels = np.asarray(movement_labels)
    surface_labels = np.asarray(surface_labels)
    speed_labels = np.asarray(speed_labels)
    return data_all_cond, movement_labels, surface_labels, speed_labels


def main():
    parser = argparse.ArgumentParser(description='Prepare a dictionary for the HAPTIX Dataset')
    parser.add_argument('--workplace', '-w', type=str, default='dis')
    args = parser.parse_args()

    if args.workplace == "dis":
        path_root = "../Datasets/HAPTIXData/1200/*.mat"
    elif args.workplace == "mac":
        path_root = "../Datasets/HAPTIXData/1200/*.mat"
    else:
        raise RuntimeError('Invalid workplace choice.')

    data_files = list(map(loadmat, glob.glob(path_root)))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Process the list of files, but split the work across the process pool to use all CPUs.
        # executor.map() function takes in the helper function to call and the list of data to process with it.
        # executor.map() function returns results in  the same order as the list of data given to the process.
        data, movement_labels, surface_labels, speed_labels = [], [], [], []
        for subject in tqdm(executor.map(extract_data, data_files), total=len(data_files)):
            data.append(subject[0])
            movement_labels.append(subject[1])
            surface_labels.append(subject[2])
            speed_labels.append(subject[3])

    # Select and pre-process EEG data here
    eeg_data_list = []
    mov_label_list = []
    surf_label_list = []
    spd_label_list = []
    for sub in range(len(data)):
        print('Subject ' + str(sub + 1) + '...')
        eeg_data = data[sub]
        mov_label = movement_labels[sub] + 1
        surf_label = surface_labels[sub] + 1
        spd_label = speed_labels[sub] + 1
        misc_label = (mov_label + 10) * surf_label  # some arbitrary coding to generate unique labels per combination

        # Subsample based on movement and surface label combinations (use the minimum number of trials available)
        misc_indices = [np.where(misc_label == m)[0] for m in np.unique(misc_label)]
        num_min_trials = np.amin([len(misc_indices[i]) for i in range(len(misc_indices))])
        keep_trials = [list(misc_indices[m][:num_min_trials]) for m in range(len(misc_indices))]
        keep_trials = [item for sublist in keep_trials for item in sublist]

        eeg_data = eeg_data[keep_trials]
        mov_label = mov_label[keep_trials]
        surf_label = surf_label[keep_trials]
        spd_label = spd_label[keep_trials]

        # Resample to one fourth of sampling rate (Fs_new=300Hz)
        eeg_data = resample(eeg_data, int(eeg_data.shape[2] // int(fs_orig//fs_new)), axis=-1)

        # Append trial data with appropriate labels
        eeg_data_list.append(eeg_data)
        mov_label_list.append(mov_label)
        surf_label_list.append(surf_label)
        spd_label_list.append(spd_label)

        print('Trials of subject ' + str(sub + 1) + ' are added...')

    HAPTIXDict = {'eeg': eeg_data_list,
                  'move': mov_label_list,
                  'surface': surf_label_list,
                  'speed': spd_label_list}

    pickle.dump(HAPTIXDict, open("HaptixDict1200.p", "wb"))


if __name__ == '__main__':
    main()
