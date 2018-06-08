import numpy as np
import pickle
import argparse
import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../')

FLAGS = None

def generate_train_test_set(self):
    classes = 30
    nrof_features = 2048

    with open('features_AID.pkl', 'rb') as f:
        features = pickle.load(f)

    train_data = np.empty((0, nrof_features))
    train_single_labels = np.empty((0,))

    test_data = np.empty((0, nrof_features))
    test_single_labels = np.empty((0,))

    features_ar = np.empty((0, nrof_features))
    labels_ar = np.empty((0,))

    print('Generating train and test set...')
    for idx in range(len(features)):
        feature = features[idx][0]
        label = features[idx][1]

        features_ar = np.append(features_ar, np.reshape(feature, newshape=(-1, nrof_features)), axis=0)
        labels_ar = np.append(labels_ar, [label,], axis=0)

    for idx_class in range(classes):
        class_label_idxs = np.where(labels_ar == idx_class)[0]
        nrof_samples_in_class = len(class_label_idxs)
        nrof_train_samples = int(np.ceil(FLAGS.train_test_split * nrof_samples_in_class))

        np.random.shuffle(class_label_idxs)

        # train data
        class_features_train = features_ar[class_label_idxs[:nrof_train_samples], :]
        class_labels_train = labels_ar[class_label_idxs[:nrof_train_samples], ]

        # test data
        class_features_test = features_ar[class_label_idxs[nrof_train_samples:], :]
        class_labels_test = labels_ar[class_label_idxs[nrof_train_samples:],]

        # append train data
        train_data = np.append(train_data, class_features_train, axis=0)
        train_single_labels = np.append(train_single_labels, class_labels_train, axis=0)

        # append test data
        test_data = np.append(test_data, class_features_test, axis=0)
        test_single_labels = np.append(test_single_labels, class_labels_test, axis=0)

    # Shuffle train data
    np.random.seed(None)
    st0 = np.random.get_state()
    np.random.shuffle(train_data)

    np.random.set_state(st0)
    np.random.shuffle(train_single_labels)

    out_dir = 'data'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # save the numpy arrays
    np.save('data/train_data.npy', train_data)
    np.save('data/train_single_labels.npy', train_single_labels)

    np.save('data/test_data.npy', test_data)
    np.save('data/test_single_labels.npy', test_single_labels)
    print('Files saved!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_test_split',
        type=float,
        default=0.6,
        help="Train test split ratio."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=generate_train_test_set, argv=[sys.argv[0]] + unparsed)

    generate_train_test_set()







