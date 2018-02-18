"""Loads a sample video and classifies using a trained Kinetics checkpoint."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import sonnet as snt
import i3d

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' 

_IMAGE_SIZE = 224
_NUM_CLASSES = 400
_SAMPLE_VIDEO_FRAMES = 25
_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

rgb_dir = "/media/SeSaMe_NAS/data/Lixinhui/I3D/data/MCAD_rgb/"
flow_dir = "/media/SeSaMe_NAS/data/Lixinhui/I3D/data/MCAD_flow/"
save_rgb_dir = "/media/SeSaMe_NAS/data/Lixinhui/I3D/feature/MCAD_rgb/"
save_flow_dir = "/media/SeSaMe_NAS/data/Lixinhui/I3D/feature/MCAD_flow/"
save_joint_dir = "/media/SeSaMe_NAS/data/Lixinhui/I3D/feature/MCAD_joint/"
def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type
  imagenet_pretrained = FLAGS.imagenet_pretrained
  if eval_type not in ['rgb', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')
  with tf.device('/gpu:0'):
      if eval_type in ['rgb', 'joint']:
    # RGB input has 3 channels.
        rgb_input = tf.placeholder(
            tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
        with tf.variable_scope('RGB'):
          rgb_model = i3d.InceptionI3d(
              _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
          rgb_logits, _ = rgb_model(
              rgb_input, is_training=False, dropout_keep_prob=1.0)
        rgb_variable_map = {}
        for variable in tf.global_variables():
          if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        rgb_net = tf.nn.avg_pool3d(rgb_logits, ksize=[1, 2, 7, 7, 1],strides=[1, 1, 1, 1, 1], padding=snt.VALID)
      if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
        flow_input = tf.placeholder(
            tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
        with tf.variable_scope('Flow'):
          flow_model = i3d.InceptionI3d(
              _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
          flow_logits, _ = flow_model(
              flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
          if variable.name.split('/')[0] == 'Flow':
            flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
        flow_net = tf.nn.avg_pool3d(flow_logits, ksize=[1, 2, 7, 7, 1],strides=[1, 1, 1, 1, 1], padding=snt.VALID)
      if eval_type == 'rgb':
        model_logits = rgb_net
      elif eval_type == 'flow':
        model_logits = flow_net
      else:
        model_logits = rgb_net + flow_net

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    feed_dict = {}
    
    if eval_type in ['rgb', 'joint']:
      if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
      tf.logging.info('RGB checkpoint restored')
      

    if eval_type in ['flow', 'joint']:
      if imagenet_pretrained:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
      else:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      tf.logging.info('Flow checkpoint restored')


    if eval_type in ['rgb']:
      for ID_file in os.listdir(rgb_dir):
        ID_file = 'ID0003'
        class_path = rgb_dir + ID_file
        for video_name in os.listdir(class_path):
            video_path = class_path +'/'+ video_name
            rgb_sample = np.load(video_path)
            rgb_sample = np.reshape(rgb_sample,(1, 25, 224, 224, 3))
            feed_dict[rgb_input] = rgb_sample
            out_logits = sess.run(model_logits,feed_dict=feed_dict)
            print (out_logits.shape)
            data_save = np.reshape(out_logits,(3,1024))
            save_rgb_path = save_rgb_dir + video_name
            np.save(save_rgb_path,data_save)
            print('save_data: %s' % save_rgb_path)
    
    if eval_type in ['flow']:
      for ID_file in os.listdir(flow_dir):
        class_path = flow_dir + ID_file
        for video_name in os.listdir(class_path):
            video_path = class_path +'/'+ video_name
            flow_sample = np.load(video_path)
            flow_sample = np.reshape(flow_sample,(1, 25, 224, 224, 2))
            feed_dict[flow_input] = flow_sample
            out_logits = sess.run(model_logits,feed_dict=feed_dict)
            out_logits.shape
            data_save = np.reshape(out_logits,(2,1024))
            save_flow_path = save_flow_dir + video_name
            np.save(save_rgb_path,data_save)
            print('save_data: %s' % save_flow_path)
    
    if eval_type in ['joint']:
      for ID_file in os.listdir(flow_dir):
        class_path = flow_dir + ID_file
        for video_name in os.listdir(class_path):
            video_path_flow = class_path +'/'+ video_name
            video_path_rgb = rgb_dir + ID_file + '/'+ video_name
            flow_sample = np.load(video_path_flow)
            rgb_sample = np.load(video_path_rgb)
            rgb_sample = np.reshape(rgb_sample,(1, 25, 224, 224, 3))
            flow_sample = np.reshape(flow_sample,(1, 25, 224, 224, 2))
            feed_dict[flow_input] = flow_sample
            feed_dict[rgb_input] = rgb_sample
            out_logits = sess.run(model_logits,feed_dict=feed_dict)
            out_logits.shape
            data_save = np.reshape(out_logits,(5,1024))
            save_joint_path = save_joint_dir + video_name
            np.save(save_joint_path,data_save)
            print('save_data: %s' % save_joint_path)
    print('OK')
    

if __name__ == '__main__':
  tf.app.run(main)
