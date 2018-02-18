
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import i3d
import random
import os
import sonnet as snt

os.environ['TF_CPP_MIN_LOG_LEVEL']='3' 

save_model_path = '/media/SeSaMe_NAS/data/Lixinhui/I3D/feature/model/mcad/rgb_fv3_iter2018.ckpt'
save_model_path2 = '/media/SeSaMe_NAS/data/Lixinhui/I3D/feature/model/mcad/rgb_fv3_epoch5.ckpt'
data_dir = '/media/SeSaMe_NAS/data/Lixinhui/I3D/data/MCAD_rgb'

_IMAGE_SIZE = 224
_NUM_CLASSES = 18
_SAMPLE_VIDEO_FRAMES =64

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

epoch_num = 5
batch_size = 8
Batch_size_test = 8
#learning_rate = 0.002
#ID0030_D_Cam04_A09_S03.npy
def generator_data(data_path_dir,fv):
    train_list=['ID0005','ID0007','ID0008','ID0012','ID0015','ID0017','ID0019','ID0020','ID0023','ID0026','ID0030','ID0032']
    if (fv =='1'):
        view_list =['Cam04','Cam05','Cam06']
    elif fv =='2':
        view_list =['Cam04','PTZ04','Cam06']
    else:
        view_list =['Cam04','PTZ04','PTZ06']
    video_list = []
    data_array = np.zeros((batch_size,_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    label_array = np.zeros((batch_size,_NUM_CLASSES))
    for ID in train_list :
        data_path = data_path_dir + '/' +ID
        for video_name in os.listdir(data_path):
            v_view = video_name[9:14]
#            v_sub = video_name[0:6]
            if v_view in view_list:
#            if v_sub in train_list:
                video_list.append(video_name)
    random.shuffle(video_list)
    lenth = len(video_list) 
    Interval = int(lenth/batch_size)
    for x in range(Interval):
        video_path = video_list[x*batch_size:(x+1)*batch_size]
        for index,name in enumerate(video_path):
            v_sub = name[0:6]
            data_array[index,:,:,:,:]= np.load(data_path_dir+'/'+ v_sub +'/'+name)
            label_array[index,:]= tf.one_hot(int(name[16:18])-1,_NUM_CLASSES).eval()
        yield data_array, label_array

def test_data(data_path_dir,Cam):
    test_data= np.zeros((8,_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    test_label= np.zeros((8,_NUM_CLASSES))
    test_list =['ID0001','ID0003','ID0004','ID0013','ID0014','ID0016','ID0018','ID0027']
#    test_list =['ID0001']
    n = 0
    for ID in test_list:
        data_path = data_path_dir +'/' + ID
        for video_name in os.listdir(data_path):
            v_view = video_name[9:14]
            v_sub = video_name[0:6]
            if n < 8:
                if v_view == Cam:
#                if v_sub in test_list:
                    test_data[n,:,:,:,:]= np.load(data_path+'/'+video_name)
                    test_label[n,:]= tf.one_hot(int(video_name[16:18])-1,_NUM_CLASSES).eval()
                    n = n+1
    return test_data,test_label


def prepare_vali_data(data_path_dir,Cam):
    test_list =['ID0001','ID0003','ID0004','ID0013','ID0014','ID0016','ID0018','ID0027']
    video_list = []
    for ID in test_list :
        data_path = data_path_dir + '/' +ID
        for video_name in os.listdir(data_path):
            v_view = video_name[9:14]
            if  v_view == Cam:
                video_list.append(video_name)
    return video_list
def generator_vali_data(video_list,data_path_dir):
    test_data= np.zeros((Batch_size_test,_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    test_label= np.zeros((Batch_size_test,_NUM_CLASSES))
    random.shuffle(video_list)
    length = len(video_list)
    index_random = random.randint(1,length-32)
    video_path = video_list[index_random:index_random+Batch_size_test]
    for index,name in enumerate(video_path):
        v_sub = name[0:6]
        test_data[index,:,:,:,:]= np.load(data_path_dir+'/'+ v_sub +'/'+name)
        test_label[index,:]= tf.one_hot(int(name[16:18])-1,_NUM_CLASSES).eval()
    return test_data,test_label

def accuracy(predictions, labels):
    return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def main(unused_argv):
    with tf.device('/gpu:1'):
        x_rgb = tf.placeholder(tf.float32, shape=(None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
        y_rgb = tf.placeholder(tf.float32, shape=(None, _NUM_CLASSES))
        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
            rgb_net, A = rgb_model(x_rgb, is_training=False, dropout_keep_prob=1.0)
            end_point = 'Logits'
            with tf.variable_scope('inception_i3d'):
                with tf.variable_scope(end_point):
                    rgb_net = tf.nn.avg_pool3d(rgb_net, ksize=[1, 2, 7, 7, 1],
                               strides=[1, 1, 1, 1, 1], padding=snt.VALID)
                    if TRAINING:
                        rgb_net = tf.nn.dropout(rgb_net, 0.7)           
                    logits = i3d.Unit3D(output_channels=_NUM_CLASSES,
                                kernel_shape=[1, 1, 1],
                                activation_fn=None,
                                use_batch_norm=False,
                                use_bias=True,
                                name='Conv3d_0c_1x1')(rgb_net, is_training=True)
        logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        averaged_logits = tf.reduce_mean(logits, axis=1)

        rgb_variable_map = {}
        last_layer =[]
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
            	if (variable.name.find("Logits")==-1):
                    rgb_variable_map[variable.name.replace(':0', '')] = variable
                    if (variable.name.find("Mixed_5c")==-1):
                        last_layer.append(variable.name.replace(':0', ''))
                        #[variable.name.replace(':0', '')] = variable
                else:
                    #last_layer[variable.name.replace(':0', '')] = variable
                    last_layer.append(variable.name.replace(':0', ''))
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

        last_layer_var  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='RGB/inception_i3d/Logits')
        for variabel_name in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='RGB/inception_i3d/Mixed_5c'):
            last_layer_var.append(variabel_name)
        print(last_layer_var)
        global_step = tf.Variable(0,trainable=False)  
        learning_rate = tf.train.exponential_decay(0.002,global_step,100,0.98,staircase=False)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= averaged_logits,labels=y_rgb)
        loss = tf.reduce_mean(cross_entropy)
        optimizer_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss,var_list=last_layer_var)
        predictions = tf.nn.softmax(averaged_logits)

        model_saver = tf.train.Saver()
        init = tf.global_variables_initializer() 

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
        tf.logging.info('RGB checkpoint restored %s',_CHECKPOINT_PATHS['rgb_imagenet'])
        i = 0
        vali_list_prepare = prepare_vali_data(data_dir,'Cam04')
        T_batch,t_batch = generator_vali_data(vali_list_prepare,data_dir)
        for epoch in range(epoch_num):
            for X_batch, y_batch in generator_data(data_dir,'3'):
                i = i+1
                feed_dict_train = {x_rgb:X_batch,y_rgb:y_batch}
                op_, loss_,pre_train = sess.run([optimizer_op, loss, predictions], feed_dict=feed_dict_train)
                train_acc= accuracy(pre_train,y_batch)
                print("After %d epoch %d training step(s), train accuracy using average model is %g " % (epoch,i, train_acc))
                print("Train_loss is  %f " % (loss_*1.0))
                if(i%100 == 0):
                    T_batch,t_batch = generator_vali_data(vali_list_prepare,data_dir)
                    feed_dict_test = {x_rgb:T_batch,y_rgb:t_batch}
                    pre_test = 	sess.run(predictions,feed_dict = feed_dict_test)
                    test_acc = train_acc= accuracy(pre_test,t_batch)
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print ("After %d iter, validate accuracy using average model is %g " % (epoch, test_acc))
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    model_saver.save(sess,save_model_path)
            model_saver.save(sess,save_model_path2)
        print("OK!!!")

if __name__ == '__main__':
    TRAINING = True
    tf.app.run(main)
