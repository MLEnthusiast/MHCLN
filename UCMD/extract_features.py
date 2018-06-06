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
import random
import argparse
import sys

# specify the GPU where you want to run the code
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#model_dir = '/data0/subhankar/models/tutorials/image/imagenet'
#images_dir = '/data0/subhankar/UC_Triplet/UCMerced_LandUse/Images'
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
    class_dict = {'agricultural': 0,
                  'airplane': 1,
                  'baseballdiamond': 2,
                  'beach': 3,
                  'buildings': 4,
                  'chaparral': 5,
                  'denseresidential': 6,
                  'forest': 7,
                  'freeway': 8,
                  'golfcourse': 9,
                  'harbor': 10,
                  'intersection': 11,
                  'mediumresidential': 12,
                  'mobilehomepark': 13,
                  'overpass': 14,
                  'parkinglot': 15,
                  'river': 16,
                  'runway': 17,
                  'sparseresidential': 18,
                  'storagetanks': 19,
                  'tenniscourt': 20}

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
            label_name = str.split(rgb_f, '.')[0]
            label = label_name[:-2]
            class_label = class_dict.get(label)


            # Save the file in jpg format
            img = Image.open(filename)
            img.thumbnail(img.size)
            filename_jpg = label_name+'.jpg'
            save_path = os.path.join(FLAGS.dump_dir, filename_jpg)
            img.save(save_path, "JPEG", quality=100)
            feature_out = extract_feature(image_filename=save_path)
            os.remove(save_path) # delete the file after extracting feature

            # append in the list with order
            # [feature, class_label, image_path]
            features.append([feature_out, class_label, filename])
    return features

def main_graph(self):
    # Creates graph from saved GraphDef.
    create_graph()

    extracted_features = generate_test_train_set(root_rgb=FLAGS.images_dir)

    # dump into a pickle file
    with open('features.pkl', 'wb') as f:
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
