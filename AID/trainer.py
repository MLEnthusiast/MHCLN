import os, sys
sys.path.append(os.getcwd())
sys.path.append('../')
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import argparse
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.plot
import pickle
from PIL import Image

# specify the GPU where you want to run the code
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = None

NUM_FEATURES = 2048
NUM_CHANNELS = 3
IMG_SIZE = 64
CLASSES = 30  # Number of classes in AID
beta = 0.001
gamma = 1

lib.print_model_settings(locals().copy())

train_data = np.load('./data/train_data.npy')
train_labels = np.load('./data/train_single_labels.npy')

test_data = np.load('./data/test_data.npy')
test_labels = np.load('./data/test_single_labels.npy')

def get_triplets(training_samples, training_labels, nrof_classes, nrof_items_per_class):

    triplets = np.empty((0, NUM_FEATURES))
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    pos_class_idx = class_indices[0]
    neg_class_idx = class_indices[1]

    pos_samples_idx = np.where(training_labels == pos_class_idx)[0]
    neg_samples_idx = np.where(training_labels == neg_class_idx)[0]

    np.random.shuffle(pos_samples_idx)
    np.random.shuffle(neg_samples_idx)

    anchors = training_samples[pos_samples_idx[0:nrof_items_per_class], :]
    positives = training_samples[pos_samples_idx[nrof_items_per_class:2*nrof_items_per_class], :]
    negatives = training_samples[neg_samples_idx[0:nrof_items_per_class], :]

    triplets = np.append(triplets, anchors, axis=0)
    triplets = np.append(triplets, positives, axis=0)
    triplets = np.append(triplets, negatives, axis=0)

    return triplets

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def main_network(inputs):

    output = tf.reshape(inputs, shape=(-1, NUM_FEATURES, 1, 1))
    output = lib.ops.conv2d.Conv2D(name='Classifier.Input', input_dim=NUM_FEATURES, output_dim=1024, filter_size=1, inputs=output, stride=1)
    output = LeakyReLU(output)
    output = lib.ops.conv2d.Conv2D(name='Classifier.2', input_dim=1024, output_dim=512, filter_size=1, inputs=output, stride=1)
    output = LeakyReLU(output)
    output = lib.ops.conv2d.Conv2D(name='Classifier.3', input_dim=512, output_dim=FLAGS.HASH_BITS, filter_size=1, inputs=output, stride=1)
    output_sigmoid = tf.nn.sigmoid(output)

    return tf.reshape(output_sigmoid, shape=[-1, FLAGS.HASH_BITS])

def triplet_loss(embeddings_0, anchor, positive, negative, alpha):
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.), 0)

        # We want to maximize this loss to encourage activations to be close to 0 or 1
        loss_2 = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(embeddings_0, 0.5 * tf.ones_like(embeddings_0))), 1))

        # We want to minimize this loss to ensure that the output of each node has nearly 50% chance of being 0 or 1
        loss_3 = tf.reduce_mean(tf.square(tf.reduce_mean(embeddings_0, 1) - 0.5))

        combined_loss = loss - beta*loss_2/FLAGS.HASH_BITS + gamma*loss_3

    return combined_loss

def train(self):
    all_samples = tf.placeholder(tf.float32, shape=[None, NUM_FEATURES])
    sigmoid_activations = main_network(all_samples)
    embeddings = tf.nn.l2_normalize(sigmoid_activations, 1, 1e-10, name='embeddings')
    anchors, positives, negatives = tf.split(embeddings, 3, axis=0)
    loss = triplet_loss(sigmoid_activations, anchors, positives, negatives, FLAGS.ALPHA)
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([loss] + regularization_loss, name='total_loss')

    train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(total_loss, var_list=lib.params_with_name('Classifier'))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for iters in range(FLAGS.ITERS):

            start_time = time.time()
            triplets = get_triplets(train_data, train_labels, CLASSES, FLAGS.BATCH_SIZE/3)
            _, cost = session.run([train_op, total_loss], feed_dict={all_samples: triplets})

            lib.plot.plot('Triplet Loss', cost)
            lib.plot.plot('Time', time.time() - start_time)

            if (iters < 10) or (iters % 100 == 99):
                lib.plot.flush()

            lib.plot.tick()

        print('Optimization finished!')

        with open('features_AID.pkl', 'rb') as f:
            features = pickle.load(f)

        test_images = np.empty((0, IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
        test_embeddings = np.empty((0, FLAGS.HASH_BITS)).astype(np.int8)
        test_single_labels = np.empty((0,))

        features_ar = np.empty((0, NUM_FEATURES))
        labels_ar = np.empty((0,))
        paths = []

        if not os.path.exists('dump_dir/'):
            os.mkdir('dump_dir/')

        print('Preparing test embeddings...')

        for idx in range(len(features)):
            feature = features[idx][0]
            label = features[idx][1]
            path = features[idx][2]

            features_ar = np.append(features_ar, np.reshape(feature, newshape=(-1, NUM_FEATURES)), axis=0)
            labels_ar = np.append(labels_ar, [label, ], axis=0)
            paths.append(path)

        start_idx = 0
        for idx_class in range(CLASSES):
            class_label_idxs = np.where(labels_ar == idx_class)[0]
            nrof_samples_in_class = len(class_label_idxs)
            nrof_train_samples = int(np.ceil(FLAGS.train_test_split * nrof_samples_in_class))
            nrof_test_samples = nrof_samples_in_class - nrof_train_samples
            end_idx = start_idx + nrof_samples_in_class

            # test data
            class_features_test = features_ar[class_label_idxs[nrof_train_samples:], :]
            class_labels_test = labels_ar[class_label_idxs[nrof_train_samples:]]
            class_paths_test = []
            for _, idx in enumerate(class_label_idxs[nrof_train_samples:]):
                class_paths_test.append(paths[idx])

            for i in range(nrof_test_samples):
                test_input = class_features_test[i,:]
                test_single_label = class_labels_test[i]
                test_img_path = class_paths_test[i]

                # Store the embedding
                test_embedding = session.run(sigmoid_activations, feed_dict={all_samples: np.reshape(test_input, newshape=(-1, NUM_FEATURES))})
                test_embedding = ((np.sign(test_embedding - 0.5) + 1) / 2)
                test_embeddings = np.append(test_embeddings, test_embedding.astype(np.int8), axis=0)

                # Store the label
                test_single_labels = np.append(test_single_labels, [test_single_label, ], axis=0)

                # Store the image
                img = Image.open(test_img_path)
                img = img.resize([IMG_SIZE, IMG_SIZE], Image.ANTIALIAS)
                img.save(os.path.join('dump_dir', 'temp.jpg'))

                read_image_path = os.path.join('dump_dir', 'temp.jpg')
                img = np.array(Image.open(read_image_path), dtype=int) / 256.
                img = np.reshape(img, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
                test_images = np.append(test_images, img, axis=0)
            start_idx = end_idx

        out_dir = 'data/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        np.save('data/test_labels.npy', test_single_labels)
        np.save('data/test_embeddings.npy', test_embeddings)
        np.save('data/test_images.npy', test_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--HASH_BITS',
        type=int,
        default=32,
        help="Hash-bit length (K)"
    )
    parser.add_argument(
        '--ALPHA',
        type=float,
        default=0.2,
        help="The alpha separation between the positive and negative samples"
    )
    parser.add_argument(
        '--BATCH_SIZE',
        type=int,
        default=90,
        help="The number of samples in a mini-batch of triplets. Must be divisible by 3."
    )
    parser.add_argument(
        '--ITERS',
        type=int,
        default=10000,
        help="Number of training iterations."
    )
    parser.add_argument(
        '--train_test_split',
        type=float,
        default=0.6,
        help="Train test split ratio."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)

    train()



