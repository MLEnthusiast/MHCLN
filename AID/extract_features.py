import os
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pickle
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import random
import sys

# specify the GPU where you want to run the code
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#model_dir = '/media/sroy/models/tutorials/image/imagenet'
#images_dir = '/media/sroy/AID'
#dump_dir = 'dump_dir/'

FLAGS = None

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""

  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def extract_feature(image_filename):
    image_data = tf.gfile.FastGFile(image_filename, 'rb').read()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        last_but_one_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        feature = sess.run(last_but_one_tensor, {'DecodeJpeg/contents:0': image_data})
        feature = np.squeeze(feature)
    return feature

def generate_test_train_set(root_rgb):
    class_dict = {'airport': 0,
                  'bareland': 1,
                  'baseballfield': 2,
                  'beach': 3,
                  'bridge': 4,
                  'center': 5,
                  'church': 6,
                  'commercial': 7,
                  'denseresidential': 8,
                  'desert': 9,
                  'farmland': 10,
                  'forest': 11,
                  'industrial': 12,
                  'meadow': 13,
                  'mediumresidential': 14,
                  'mountain': 15,
                  'park': 16,
                  'parking': 17,
                  'playground': 18,
                  'pond': 19,
                  'port': 20,
                  'railwaystation': 21,
                  'resort': 22,
                  'river': 23,
                  'school': 24,
                  'sparseresidential': 25,
                  'square': 26,
                  'stadium': 27,
                  'storagetanks': 28,
                  'viaduct': 29}

    if not os.path.exists(FLAGS.dump_dir):
        os.mkdir(FLAGS.dump_dir)

    # data structure
    features = []

    _, dirs, _ = os.walk(root_rgb).next()

    for d in dirs:
        print('Reading %s' % d)
        img_dir = os.path.join(root_rgb, d)
        _, _, rgb_files = os.walk(img_dir).next()

        random.shuffle(rgb_files)

        for rgb_f in rgb_files:
            filename = os.path.join(img_dir, rgb_f)
            print('Reading %s' % rgb_f)

            # Read the labels
            label_name = str.split(rgb_f, '_')[0]
            class_label = class_dict.get(label_name)

            # read the raw image
            feature_out = extract_feature(image_filename=filename)

            # append in the list
            # [feature, class_label, image_path]
            features.append([feature_out, class_label, filename])
    return features

def main_graph(self):
    # Creates graph from saved GraphDef.
    create_graph()

    extracted_features = generate_test_train_set(root_rgb=FLAGS.images_dir)

    # dump into a pickle
    with open('features_AID.pkl', 'wb') as f:
        pickle.dump(extracted_features, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/imagenet',
        required=True,
        help="""\
            Specify the absolute path where the pre-trained Inception model is downloaded. 
            Details for downloading tensorflow pre-trained model from this link: 
            https://www.tensorflow.org/tutorials/image_recognition\
            """
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        default='',
        required=True,
        help="""\
            Specify the absolute path of the data set\
        """
    )
    parser.add_argument(
        '--dump_dir',
        type=str,
        default='dump_dir/',
        help="""\
                The directory to store temporary files\
            """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main_graph, argv=[sys.argv[0]] + unparsed)
    main_graph()
