
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import i3d
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' 

save_model_path = '/media/SeSaMe_NAS/data/Lixinhui/I3D/feature/model/mcad/rgb_fv2_epoch3.ckpt'
data_dir = '/media/SeSaMe_NAS/data/Lixinhui/I3D/feature/MCAD_rgb'

_IMAGE_SIZE = 224
_NUM_CLASSES = 18

epoch_num = 3
batch_size = 50
learning_rate = 0.002
#ID0030_D_Cam04_A09_S03.npy
def generator_data(data_path,fv):
    train_list=['ID0005','ID0007','ID0008','ID0012','ID0015','ID0017','ID0019','ID0020','ID0023','ID0026','ID0030','ID0032']
    if (fv =='1'):
        view_list =['Cam04','Cam05','Cam06']
    elif fv =='2':
        view_list =['Cam04','PTZ04','Cam06']
    else:
        view_list =['Cam04','PTZ04','PTZ06']
    video_list = []
    data_array = np.zeros((batch_size,3,1,1,1024))
    label_array = np.zeros((batch_size,_NUM_CLASSES))
    for video_name in os.listdir(data_path):
        v_view = video_name[9:14]
        v_sub = video_name[0:6]
        if v_view in view_list:
            if v_sub in train_list:
                video_list.append(video_name)
    random.shuffle(video_list)
    lenth = len(video_list) 
    Interval = int(lenth/batch_size)
    for x in range(Interval):
        video_path = video_list[x*batch_size:(x+1)*batch_size]
        for index,name in enumerate(video_path):
            data_array[index,:,0,0,:]= np.load(data_path+'/'+name)
            label_array[index,:]= tf.one_hot(int(name[16:18])-1,_NUM_CLASSES).eval()
        yield data_array, label_array

def test_data(data_path,Cam):
    test_data= np.zeros((200,3,1,1,1024))
    test_label= np.zeros((200,_NUM_CLASSES))
    test_list =['ID0001','ID0003','ID0004','ID0013','ID0014','ID0016','ID0018','ID0027']
    n = 0
    for video_name in os.listdir(data_path):
        v_view = video_name[9:14]
        v_sub = video_name[0:6]
        if n < 200:
            if v_view == Cam:
                if v_sub in test_list:
                    test_data[n,:,0,0,:]= np.load(data_path+'/'+video_name)
                    test_label[n,:]= tf.one_hot(int(video_name[16:18])-1,_NUM_CLASSES).eval()
                    n = n+1
    return test_data,test_label

def accuracy(predictions, labels):
    return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def main(unused_argv):
    with tf.device('/gpu:2'):
        x_input = tf.placeholder(tf.float32, shape=(None, 3, 1,1,1024))
        y_ = tf.placeholder(tf.float32, shape=(None, _NUM_CLASSES))
        end_point = 'Logits'
        with tf.variable_scope(end_point):
            if TRAINING:
                x_input = tf.nn.dropout(x_input, 0.7)
            logits = i3d.Unit3D(output_channels=_NUM_CLASSES,
                        kernel_shape=[1, 1, 1],
                        activation_fn=None,
                        use_batch_norm=False,
                        use_bias=True,
                        name='Conv3d_0c_1x1')(x_input, is_training=True)
        logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        averaged_logits = tf.reduce_mean(logits, axis=1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= averaged_logits,labels=y_)
        loss = tf.reduce_mean(cross_entropy)
        optimizer_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)
            
        predictions = tf.nn.softmax(averaged_logits)
        model_saver = tf.train.Saver()
        init = tf.global_variables_initializer() 

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
        i = 0
        T_batch,t_batch = test_data(data_dir,'Cam04')
        feed_dict_test = {x_input:T_batch,y_:t_batch}
        for epoch in range(epoch_num):
            for X_batch, y_batch in generator_data(data_dir,'2'):
                i = i+1
                feed_dict_train = {x_input:X_batch,y_:y_batch}
                op_, loss_,pre_train = sess.run([optimizer_op, loss, predictions], feed_dict=feed_dict_train)
                train_acc= accuracy(pre_train,y_batch)
                print("After %d epoch %d training step(s), train accuracy using average model is %g " % (epoch,i, train_acc))
                print("Train_loss is  %f " % (loss_*1.0))

            pre_test = 	sess.run(predictions,feed_dict = feed_dict_test)
            test_acc = train_acc= accuracy(pre_test,t_batch)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print ("After %d epoch, validate accuracy using average model is %g " % (epoch, test_acc))
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            model_saver.save(sess,save_model_path)
        print("OK!!!")

if __name__ == '__main__':
    TRAINING = True
    tf.app.run(main)
