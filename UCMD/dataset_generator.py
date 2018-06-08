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

    nrof_train_per_class = int(round(FLAGS.train_test_split * 100))
    nrof_samples_per_class = 100 # UCMD has 100 images per class
    classes = 21 # UCMD has 21 classes
    nrof_features = 2048

    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)

    train_data = np.empty((0, nrof_features))
    train_single_labels = np.empty((0,))

    test_data = np.empty((0, nrof_features))
    test_single_labels = np.empty((0,))

    start_idx = 0
    for idx_class in range(classes):
        end_idx = (idx_class + 1) * nrof_samples_per_class
        class_features = features[start_idx:end_idx]

        train_class_features = class_features[:nrof_train_per_class]
        test_class_features = class_features[nrof_train_per_class:]

        for idx in range(len(train_class_features)):
            train_datum = train_class_features[idx][0]
            train_single_label = train_class_features[idx][1]

            train_data = np.append(train_data, np.reshape(train_datum, newshape=(-1, nrof_features)), axis=0)
            train_single_labels = np.append(train_single_labels, [train_single_label, ], axis=0)

        for idx in range(len(test_class_features)):
            test_datum = test_class_features[idx][0]
            test_single_label = test_class_features[idx][1]

            test_data = np.append(test_data, np.reshape(test_datum, newshape=(-1, nrof_features)), axis=0)
            test_single_labels = np.append(test_single_labels, [test_single_label, ], axis=0)

        start_idx = end_idx

    # Shuffle train data
    np.random.seed(None)
    st0 = np.random.get_state()
    np.random.shuffle(train_data)

    np.random.set_state(st0)
    np.random.shuffle(train_single_labels)

    if not os.path.exists('data/'):
        os.mkdir('data/')

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









