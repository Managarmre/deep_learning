# @author Pauline Houlgatte
# based on https://www.tensorflow.org/tutorials/image_recognition

import argparse
import os
import re
import numpy as np
import tarfile as tr
import tensorflow as tf

def argparser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", default="./test_chat.tar.gz", help="name of tar file")
    parser.add_argument("-i", default="cat.jpg", help="name of the image")
    parser.add_argument("-o", default="/tmp/", help="output directory")
    parser.add_argument("-p", type=int, default=10, help="display x best predictions")
    args = parser.parse_args()
    return args

def download(file,output):
  if (os.path.exists(file)):
    tr.open(file,'r:gz').extractall(output)
    return True
  return False

def recoverImage(file,output):
  image = os.path.join(output,file)
  return image

def imageRecognition(image,output,nb_predictions):
  img = tf.gfile.FastGFile(image,'rb').read()
  # file.pb recover on tendorflow github
  # here we create a graph from .pb file
  graph = tf.GraphDef()
  graph_file = tf.gfile.FastGFile(os.path.join(output,'classify_image_graph_def.pb'))
  graph.ParseFromString(graph_file.read())
  graph_def = tf.import_graph_def(graph,name='')

  with tf.Session() as sess:
    # softmax:0 contain the normalized prediction across 1000 labels
    softmax = sess.graph.get_tensor_by_name('softmax:0')
    # DecodeJpeg/contents:0 contain JPEG encoding (format string) of the image
    prediction = sess.run(softmax,{'DecodeJpeg/contents:0': img})
    # remove single dimensional entries from the shape of an array
    prediction = np.squeeze(prediction)
    # create node ID
    # node = NodeLookup()

    # recover node id of best predictions
    top_predictions = prediction.argsort()[-nb_predictions:][::-1]
    return top_predictions,prediction

def recoverLabels(output):
  labels = {}
  label_path = os.path.join(output, 'imagenet_2012_challenge_label_map_proto.pbtxt')
  uid_path = os.path.join(output, 'imagenet_synset_to_human_label_map.txt')

  # copy from github tensorflow (image_recognition, function load)
  # Loads mapping from string UID to human-readable string
  proto_as_ascii_lines = tf.gfile.GFile(uid_path).readlines()
  uid_to_human = {}
  p = re.compile(r'[n\d]*[ \S,]*')
  for line in proto_as_ascii_lines:
    parsed_items = p.findall(line)
    uid = parsed_items[0]
    human_string = parsed_items[2]
    uid_to_human[uid] = human_string

  # Loads mapping from string UID to integer node ID.
  node_id_to_uid = {}
  proto_as_ascii = tf.gfile.GFile(label_path).readlines()
  for line in proto_as_ascii:
    if line.startswith('  target_class:'):
      target_class = int(line.split(': ')[1])
    if line.startswith('  target_class_string:'):
      target_class_string = line.split(': ')[1]
      node_id_to_uid[target_class] = target_class_string[1:-2]

  # Loads the final mapping of integer node ID to human-readable string
  for key, val in node_id_to_uid.items():
    if val not in uid_to_human:
      tf.logging.fatal('Failed to locate: %s', val)
    name = uid_to_human[val]
    labels[key] = name

  return labels

def convertPredictions(labels,prediction):
  return labels[prediction]
  
def main(_):
  arg = argparser()
  data = arg.t
  picture_name = arg.i
  output = arg.o
  nb_predictions = arg.p

  print "==== tensorflow ===="

  success = download(data,output)
  if success:
    im = recoverImage(picture_name,output)
    tp,prediction = imageRecognition(im,output,nb_predictions)
    # print tp
    lab = recoverLabels(output)
    # print lab
    for p in tp:
      tpc = convertPredictions(lab,p)
      print tpc, prediction[p]

  print "==== end of program ===="

if __name__ == '__main__':
  tf.app.run(main=main)
