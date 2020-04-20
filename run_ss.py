#!/usr/bin/env python
import argparse
import os
import numpy as np
import scipy.io as sio
import pickle
from scipy.special import softmax
from net import AdversarialCNN
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.utils import to_categorical


def main():
    parser = argparse.ArgumentParser(description='Adversarial EEG Decoding for HAPTIX')
    parser.add_argument('--subject', '-s', type=int, default=-1, help='Subject index to run (default = -1 (do all)')
    parser.add_argument('--repeat', '-r', type=int, default=1, help='CV repetition - used as a random seed')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the results')
    parser.add_argument('--architecture', '-a', type=str, default='EEGNet', help='Decoding model architecture')
    parser.add_argument('--adversarial', '-ad', action='store_true', help='Enable adversarial training of CNN')
    parser.add_argument('--adv_lam', '-lam', type=float, default=0.01, help='Weight constant for the adversarial loss')
    parser.add_argument('--load', '-l', action='store_true', help='Load pre-trained model weights')
    args = parser.parse_args()

    train_set, val_set, test_set = generate_splits(cv_seed=args.repeat)
    subject_range = range(len(train_set)) if args.subject == -1 else [args.subject]

    # Model train-validate-test for each subject
    for sub in subject_range:
        print("Current Subject Number: " + str(sub + 1))
        train_data, val_data, test_data = train_set[sub], val_set[sub], test_set[sub]

        for fold in range(len(train_data[0])):
            print("Current Fold Number: " + str(fold + 1))
            x_train, y_train, s_train = train_data[0][fold], train_data[1][fold], train_data[2][fold]
            x_val, y_val, s_val = val_data[0][fold], val_data[1][fold], val_data[2][fold]
            x_test, y_test, s_test = test_data[0][fold], test_data[1][fold], test_data[2][fold]

            num_channels = x_train.shape[1]
            num_samples = x_train.shape[2]
            num_classes = y_train.shape[1]
            num_nuisance = s_train.shape[1]

            net = AdversarialCNN(chans=num_channels, samples=num_samples, n_output=num_classes, n_nuisance=num_nuisance,
                                 architecture=args.architecture, adversarial=args.adversarial, lam=args.adv_lam)

            # Prepare data saving location
            save_loc = args.out + '_sub' + str(sub) + '_fold' + str(fold) + '_rep' + str(args.repeat)
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
            with open(save_loc + '/test_acc.csv', 'a') as f:
                f.write(str(sub) + ',' + str(fold) + ',' + str(y_acc) + ',' + str(s_acc) + '\n')


def generate_splits(cv_seed=1):
    dataDict = pickle.load(open("HaptixDict1200.p", "rb"))
    eeg_list, move_list, surface_list = dataDict['eeg'], dataDict['move'], dataDict['surface']

    # Construct train/validation/test data set splits
    train_set, val_set, test_set = [], [], []
    for sub in range(len(eeg_list)):
        subject_eeg_array = eeg_list[sub]
        subject_label_array = surface_list[sub]
        subject_nuisance_array = move_list[sub]
        subject_misc_array = (subject_nuisance_array + 10) * subject_label_array

        sub_train_eeg, sub_train_labels, sub_train_nuisance = [], [], []
        sub_val_eeg, sub_val_labels, sub_val_nuisance = [], [], []
        sub_test_eeg, sub_test_labels, sub_test_nuisance = [], [], []
        skf_test = StratifiedKFold(n_splits=5, random_state=int(cv_seed * sub), shuffle=True)
        for train_index, test_index in skf_test.split(subject_eeg_array, subject_misc_array):
            X_train, s_eeg_test = subject_eeg_array[train_index], subject_eeg_array[test_index]
            y_train, s_label_test = subject_label_array[train_index], subject_label_array[test_index]
            s_train, s_nui_test = subject_nuisance_array[train_index], subject_nuisance_array[test_index]
            l_misc, s_misc_test = subject_misc_array[train_index], subject_misc_array[test_index]

            s_eeg_train, s_eeg_val, s_label_train, s_label_val, s_nui_train, s_nui_val = \
                    train_test_split(X_train, y_train, s_train, test_size=0.125, stratify=l_misc,
                                     random_state=int(cv_seed * sub))
            sub_train_eeg.append(s_eeg_train[:, :, :, np.newaxis])
            sub_val_eeg.append(s_eeg_val[:, :, :, np.newaxis])
            sub_test_eeg.append(s_eeg_test[:, :, :, np.newaxis])
            sub_train_labels.append(to_categorical(s_label_train - 1))
            sub_val_labels.append(to_categorical(s_label_val - 1))
            sub_test_labels.append(to_categorical(s_label_test - 1))
            sub_train_nuisance.append(to_categorical(s_nui_train - 1))
            sub_val_nuisance.append(to_categorical(s_nui_val - 1))
            sub_test_nuisance.append(to_categorical(s_nui_test - 1))
        train_set.append((sub_train_eeg, sub_train_labels, sub_train_nuisance))
        val_set.append((sub_val_eeg, sub_val_labels, sub_val_nuisance))
        test_set.append((sub_test_eeg, sub_test_labels, sub_test_nuisance))

    return train_set, val_set, test_set


if __name__ == '__main__':
    main()
