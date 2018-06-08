import os, sys

sys.path.append(os.getcwd())
sys.path.append('../')
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import operator
import warnings
import argparse
warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import hamming
import seaborn as sns
sns.set()

FLAGS = None
IMG_SIZE = 64
NUM_CHANNELS = 3
CLASSES = 30
class_names = ['airport', 'bareland', 'baseball\nfield', 'beach', 'bridge', 'center',
               'church', 'commercial', 'dense\nresidential', 'desert', 'farmland', 'forest',
               'industrial', 'meadow', 'medium\nresidential', 'mountain', 'park', 'parking',
               'playground', 'pond', 'port', 'railway\nstation', 'resort', 'river', 'school',
               'sparse\nresidential', 'square', 'stadium', 'storage\ntanks', 'viaduct']

test_data = np.load('./data/test_images.npy')
test_labels = np.load('./data/test_labels.npy')
test_encodings = np.load('./data/test_embeddings.npy')

def hamming_distance(instance1, instance2):
    return hamming(instance1, instance2)

def get_k_hamming_neighbours(nrof_neighbors, enc_test, test_img, test_lab, index):
    _neighbours = []  # 1(query image) + nrof_neighbours
    distances = []
    for i in range(len(test_encodings)):
        if index != i:  # exclude the test instance itself from the search set
            dist = hamming_distance(test_encodings[i, :], enc_test)
            distances.append((test_data[i, :, :, :], test_labels[i], dist))

    distances.sort(key=operator.itemgetter(2))
    _neighbours.append((test_img, test_lab))
    for j in range(nrof_neighbors):
        _neighbours.append((distances[j][0], distances[j][1]))

    return _neighbours


def top_k_accuracy(neghbours_list, nrof_neighbors):
    test_sample_label = neghbours_list[0][0]
    true_label_vector = [test_sample_label] * nrof_neighbors
    pred_label_vector = []

    for i in range(1, nrof_neighbors + 1):
        pred_label_vector.append(neghbours_list[i][1])

    accuracy = accuracy_score(y_true=true_label_vector, y_pred=pred_label_vector)
    return accuracy

def get_mAP(neghbours_list, nrof_neighbors):
    test_sample_label = neghbours_list[0][1]
    acc = np.empty((0,)).astype(np.float)
    correct = 1
    for i in range(1, nrof_neighbors+1):
        if test_sample_label == neghbours_list[i][1]:
            precision = (correct / float(i))
            acc = np.append(acc, [precision, ], axis=0)
            correct += 1
    if correct == 1:
        return 0.
    num = np.sum(acc)
    den = correct - 1
    return num/den

def plot_or_save_singlelabel_images(_neighbours, filename=''):
    # create figure with sub-plots
    fig, axes = plt.subplots(5, 4, figsize=(6,6), squeeze=False)
    # adjust vertical spacing if we need to print ensemble and best-net
    fig.subplots_adjust(hspace=0.6, wspace=0.1)

    query_label = _neighbours[0][1]
    acc = round(get_mAP(_neighbours, FLAGS.k), 4)
    acc_str = str(acc * 100) + "%"
    for i, ax in enumerate(axes.flat):
        # Plot image
        img = np.reshape(_neighbours[i][0], newshape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
        #img = img.transpose(1, 2, 0)
        ax.imshow(img, cmap='gray')
        # Name of the true class
        cls_true_label = _neighbours[i][1]
        cls_true_name = class_names[int(cls_true_label)]

        # Show the true and predicted classes
        if i == 0:
            xlabel = "{0}".format(class_names[int(query_label)])
            #ylabel = acc_str
            ylabel = "Query"
            ax.set_ylabel(ylabel)
        else:
            # name of the predicted class
            xlabel = "({0}) {1}".format(i, cls_true_name)
        # show the classes as the label on the x-axis
        ax.set_xlabel(xlabel)
        # show the accuracy on y-axis
        # remove the ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
        if query_label != cls_true_label:
            ax.spines['left'].set_color('red')
            ax.spines['right'].set_color('red')
            ax.spines['top'].set_color('red')
            ax.spines['bottom'].set_color('red')
            ax.spines['left'].set_linewidth(3)
            ax.spines['right'].set_linewidth(3)
            ax.spines['top'].set_linewidth(3)
            ax.spines['bottom'].set_linewidth(3)

        if i == 0:
            ax.spines['left'].set_color('blue')
            ax.spines['right'].set_color('blue')
            ax.spines['top'].set_color('blue')
            ax.spines['bottom'].set_color('blue')
            ax.spines['left'].set_linewidth(3)
            ax.spines['right'].set_linewidth(3)
            ax.spines['top'].set_linewidth(3)
            ax.spines['bottom'].set_linewidth(3)

    plt.savefig( filename, bbox_inches='tight')
    return

def eval(self):
    total_map = 0.
    out_dir = 'hamming_out/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    start_time = time.time()
    for idx in range(len(test_encodings)):

        test_encoding = test_encodings[idx, :]
        test_image = test_data[idx, :, :, :]
        test_label = test_labels[idx]
        neighbours = get_k_hamming_neighbours(nrof_neighbors=FLAGS.k, enc_test=test_encoding, test_img=test_image, test_lab=test_label, index=idx)
        total_map += get_mAP(neighbours, FLAGS.k)

        if idx % FLAGS.interval == (FLAGS.interval - 1):
            print('{0} files remaining.'.format(len(test_encodings) - idx - 1))
            f_str = './hamming_out/sample_' + str(idx) + '.png'
            plot_or_save_singlelabel_images(neighbours, f_str)


    print('mAP@20:{0}'.format((total_map / len(test_encodings)) * 100))
    print('Time taken for retrieving {0} test images:{1}'.format(len(test_encodings), time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--k',
        type=int,
        default=20,
        help="Number of closest matches to the query image"
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=100,
        help="Interval of output images"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=eval, argv=[sys.argv[0]] + unparsed)

    eval()