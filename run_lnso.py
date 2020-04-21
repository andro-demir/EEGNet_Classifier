#!/usr/bin/env python
import argparse
import os
import numpy as np
import scipy.io as sio
import pickle
from scipy.special import softmax
from net import AdversarialCNN
from sklearn.model_selection import train_test_split, KFold
from keras.utils import to_categorical


def main():
    parser = argparse.ArgumentParser(description='Adversarial EEG Decoding for HAPTIX')
    parser.add_argument('--repeat', '-r', type=int, default=1, help='CV repetition - used as a random seed')
    parser.add_argument('--lnso', '-n', type=int, default=2, help='LNSO folds (e.g., 11 for LOSO, default: 2 as LHSO')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the results')
    parser.add_argument('--architecture', '-a', type=str, default='EEGNet', help='Decoding model architecture')
    parser.add_argument('--adversarial', '-ad', action='store_true', help='Enable adversarial training of CNN')
    parser.add_argument('--adv_lam', '-lam', type=float, default=0.01, help='Weight constant for the adversarial loss')
    parser.add_argument('--load', '-l', action='store_true', help='Load pre-trained model weights')
    args = parser.parse_args()

    train_set, val_set, test_set = generate_lnso_splits(cv_seed=args.repeat, num_folds=args.lnso)

    # Leave-one-subject-out CV
    for fold in range(len(train_set)):
        print("Current Fold Number: " + str(fold + 1))

        x_train, y_train, s_train, train_ids = train_set[fold]
        x_val, y_val, s_val, train_ids = val_set[fold]
        x_test, y_test, s_test, transfer_ids = test_set[fold]

        num_channels = x_train.shape[1]
        num_samples = x_train.shape[2]
        num_classes = y_train.shape[1]
        num_nuisance = s_train.shape[1]

        net = AdversarialCNN(chans=num_channels, samples=num_samples, n_output=num_classes, n_nuisance=num_nuisance,
                             architecture=args.architecture, adversarial=args.adversarial, lam=args.adv_lam)

        # Prepare data saving location
        save_loc = args.out + '_fold' + str(fold) + '_rep' + str(args.repeat)
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

        if args.load:
            net.enc.load_weights(save_loc + '/enc.h5', by_name=True)
            net.cla.load_weights(save_loc + '/cla.h5', by_name=True)
            net.adv.load_weights(save_loc + '/adv.h5', by_name=True)
        else:
            net.train((x_train, y_train, s_train), (x_val, y_val, s_val), log_file=save_loc)
            print('Model training done, saving weights into folder: ' + save_loc)
            net.enc.save_weights(save_loc + '/enc.h5')
            net.cla.save_weights(save_loc + '/cla.h5')
            net.adv.save_weights(save_loc + '/adv.h5')

        # Evaluate and save test set predictions
        y_pred = softmax(net.acnn.predict_on_batch(x_test)[0], axis=1)
        s_pred = softmax(net.acnn.predict_on_batch(x_test)[1], axis=1)
        y_acc = np.mean(np.equal(np.argmax(y_test, axis=-1), np.argmax(y_pred, axis=-1)))
        s_acc = np.mean(np.equal(np.argmax(s_test, axis=-1), np.argmax(s_pred, axis=-1)))
        sio.savemat(save_loc + '/test_outputs.mat', {'y_pred': y_pred, 'y_true': y_test,
                                                     's_pred': s_pred, 's_true': s_test})
        sio.savemat(save_loc + '/fold_info.mat', {'train_ids': train_ids, 'transfer_ids': transfer_ids})

        with open(save_loc + '/test_acc.csv', 'a') as f:
            f.write(str(fold) + ',' + str(y_acc) + ',' + str(s_acc) + '\n')


def generate_lnso_splits(cv_seed=1, num_folds=2):
    dataDict = pickle.load(open("HaptixDict1200.p", "rb"))
    eeg_list, move_list, surface_list = dataDict['eeg'], dataDict['move'], dataDict['surface']

    # z-score transformation for EEG segments - to normalize across subjects
    for i in range(len(eeg_list)):
        mean_sub = np.mean(eeg_list[i], axis=2, keepdims=True)
        std_sub = np.std(eeg_list[i], axis=2, keepdims=True)
        eeg_list[i] = ((eeg_list[i] - mean_sub) / std_sub)

    # Construct train/validation/test data set splits
    train_set, val_set, test_set = [], [], []
    subjectIDs = np.unique(range(len(eeg_list)))
    kf = KFold(n_splits=num_folds, random_state=cv_seed, shuffle=True)
    for train_index, transfer_index in kf.split(subjectIDs):
        train_ids, transfer_ids = subjectIDs[train_index], subjectIDs[transfer_index]

        sub_test_eeg, sub_test_labels, sub_test_nuisance = [], [], []
        for s in transfer_ids:
            sub_test_eeg.append(eeg_list[s][:, :, :, np.newaxis])
            sub_test_labels.append(to_categorical(surface_list[s] - 1))
            sub_test_nuisance.append(to_categorical(move_list[s] - 1))
        sub_test_eeg = np.concatenate(sub_test_eeg, axis=0)
        sub_test_labels = np.concatenate(sub_test_labels, axis=0)
        sub_test_nuisance = np.concatenate(sub_test_nuisance, axis=0)

        sub_train_eeg, sub_train_labels, sub_train_nuisance = [], [], []
        sub_val_eeg, sub_val_labels, sub_val_nuisance = [], [], []
        for s in train_ids:
            subject_eeg_array = eeg_list[s]
            subject_label_array = surface_list[s]
            subject_nuisance_array = move_list[s]
            subject_misc_array = (subject_nuisance_array + 10) * subject_label_array

            s_eeg_train, s_eeg_val, s_label_train, s_label_val, s_nui_train, s_nui_val = \
                    train_test_split(subject_eeg_array, subject_label_array, subject_nuisance_array,
                                     test_size=0.2, stratify=subject_misc_array, random_state=int(cv_seed * s))

            sub_train_eeg.append(s_eeg_train[:, :, :, np.newaxis])
            sub_val_eeg.append(s_eeg_val[:, :, :, np.newaxis])
            sub_train_labels.append(to_categorical(s_label_train - 1))
            sub_val_labels.append(to_categorical(s_label_val - 1))
            sub_train_nuisance.append(to_categorical(s_nui_train - 1))
            sub_val_nuisance.append(to_categorical(s_nui_val - 1))
        sub_train_eeg = np.concatenate(sub_train_eeg, axis=0)
        sub_val_eeg = np.concatenate(sub_val_eeg, axis=0)
        sub_train_labels = np.concatenate(sub_train_labels, axis=0)
        sub_val_labels = np.concatenate(sub_val_labels, axis=0)
        sub_train_nuisance = np.concatenate(sub_train_nuisance, axis=0)
        sub_val_nuisance = np.concatenate(sub_val_nuisance, axis=0)

        train_set.append((sub_train_eeg, sub_train_labels, sub_train_nuisance, train_ids))
        val_set.append((sub_val_eeg, sub_val_labels, sub_val_nuisance, train_ids))
        test_set.append((sub_test_eeg, sub_test_labels, sub_test_nuisance, transfer_ids))

    return train_set, val_set, test_set


if __name__ == '__main__':
    main()
