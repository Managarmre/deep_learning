# @author Pauline Houlgatte
# based on https://www.tensorflow.org/tutorials/image_recognition

import argparse
import os
import tarfile as tr
import tensorflow as tf

def argparser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", default="./test_chat.tar.gz", help="name of tar file")
    parser.add_argument("-p", default="cat.jpg", help="name of picture")
    parser.add_argument("-o", default="/tmp/", help="output directory")
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

def main(_):
  arg = argparser()
  data = arg.t
  picture_name = arg.p
  output = arg.o

  print "==== tensorflow ===="

  success = download(data,output)
  if success:
    im = recoverImage(picture_name,output)
    print im

  print "==== end of program ===="

if __name__ == '__main__':
  tf.app.run(main=main)
