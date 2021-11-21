from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
import CogAlign
from tensorflow.python.framework import ops 
import os

def model_input(batch,m,i,temp_order,train_word,train_label,train_seqlen,train_pos_feture,
           train_relative_positions,train_reverse_relative_positions,
           train_eye_feature,train_eeg_feature,is_text,task_label):
    temp_word = []
    temp_label = []
    temp_length = []
    temp_pos = []
    temp_relative_positions = []
    temp_reverse_relative_positions = []
    temp_eye_feature = []
    temp_eeg_feature = []
    temp_input_index = temp_order[i * batch:(i + 1) * batch]
    for k in range(len(temp_input_index)):
        temp_word.append(train_word[temp_input_index[k]])
        temp_label.append(train_label[temp_input_index[k]])
        temp_length.append(train_seqlen[temp_input_index[k]])
        temp_pos.append(train_pos_feture[temp_input_index[k]])
        temp_relative_positions.append(train_relative_positions[temp_input_index[k]])
        temp_reverse_relative_positions.append(train_reverse_relative_positions[temp_input_index[k]])
        temp_eye_feature.append(train_eye_feature[temp_input_index[k]])
        temp_eeg_feature.append(train_eeg_feature[temp_input_index[k]])
    feed_dict={}
    feed_dict[m.emb_input]=np.asarray(temp_word)
    feed_dict[m.label]=np.asarray(temp_label)
    feed_dict[m.sentence_len]=np.asarray(temp_length)
    feed_dict[m.pos_input]=np.asarray(temp_pos)
    feed_dict[m.relative_position_input]=np.asarray(temp_relative_positions)
    feed_dict[m.reverse_position_input]=np.asarray(temp_reverse_relative_positions)
    feed_dict[m.eye_input]=np.asarray(temp_eye_feature)
    feed_dict[m.eeg_input]=np.asarray(temp_eeg_feature)
    feed_dict[m.is_text]=is_text
    feed_dict[m.task_label]=np.asarray(task_label)
    return feed_dict
	
def main(_):
    data_path = ''
    embedding_path = ''
    model_path = ''
    is_adv = True
    is_attention = True
    cog_ = 'EEG'
    mode = 'multi'
    task = 'relation'
    embedding = np.load(embedding_path)
    fold_num=5
    for fold in range(fold_num):
        setting = CogAlign.Setting()
        train_word = np.load(data_path+'/fold_{}/train_word.npy'.format(fold))
        train_label = np.load(data_path+'data/fold_{}/train_tag_sparse.npy'.format(fold))
        train_seqlen = np.load(data_path+'data/fold_{}/train_seqlen.npy'.format(fold))
        train_pos_feture = np.load(data_path+'data/fold_{}/train_pos_feture.npy'.format(fold))
        train_relative_positions = np.load(data_path+'data/fold_{}/train_relative_positions.npy'.format(fold))
        train_reverse_relative_positions = np.load(data_path+'data/fold_{}/train_reverse_relative_positions.npy'.format(fold))
        train_eye_feature = np.load(data_path+'data/fold_{}/train_eye_feature.npy'.format(fold))
        train_eeg_feature = np.load(data_path+'data/fold_{}/train_eeg_feature.npy'.format(fold))
        sen_len = train_seqlen[0]
        task_text = [[1,0]]*setting.batch_size
        task_cognitive = [[0,1]]*setting.batch_size
        save_path = model_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
                 
        with tf.Graph().as_default():
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            with sess.as_default():  
                initializer = tf.contrib.layers.xavier_initializer()
                with tf.variable_scope('relation_model',reuse=tf.AUTO_REUSE,initializer=initializer):
                    m = CogAlign.TransferModel(setting,tf.cast(embedding,tf.float32),is_adv=is_adv,cog_signals=cog_,
                                            text_aware_att=is_attention,tags_num=11,maxSentencelen=sen_len,task=task)
                if mode == 'single':
                    m.single_task()
                elif mode == 'multi':
                    m.multi_task()
                global_step = tf.Variable(0, name="global_step", trainable=False)
                global_step1 = tf.Variable(0, name="global_step1", trainable=False)
                optimizer = tf.train.AdamOptimizer(0.001)
                if setting.clip>0:
                    grads, vs = zip(*optimizer.compute_gradients(m.loss))
                    grads,_ = tf.clip_by_global_norm(grads,clip_norm=setting.clip)
                    train_op = optimizer.apply_gradients(zip(grads,vs),global_step)
                    train_op1 = optimizer.apply_gradients(zip(grads, vs), global_step1)
                else:
                    train_op = optimizer.minimize(m.loss, global_step)
                    train_op1 = optimizer.minimize(m.loss, global_step1)

                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(max_to_keep=None)
                
                for one_epoch in range(setting.num_epoches):
                    temp_order = list(range(len(train_word)))
                    np.random.shuffle(temp_order)
                    if mode == 'single':
                        for i in range(len(temp_order)//setting.batch_size):
                            feed_dict = model_input(setting.batch_size,m,i,temp_order,train_word,
                                            train_label,train_seqlen,train_pos_feture,
                                            train_relative_positions,train_reverse_relative_positions,
                                            train_eye_feature,train_eeg_feature,1,task_text)                           
                            _, step, loss = sess.run([train_op,global_step,m.loss],feed_dict)
                            if step % 100 == 0:
                                temp = "step {},loss {}".format(step, loss/setting.batch_size)
                                print(temp)
                            current_step = step
                            if current_step % 500 == 0:
                                saver.save(sess, save_path=save_path, global_step=current_step)
                    elif mode == 'multi':
                        for i in range(len(temp_order)//setting.batch_size):
                            for j in range(2):
                                if j == 0:
                                    feed_dict = model_input(setting.batch_size,m,i,temp_order,train_word,
                                                    train_label,train_seqlen,train_pos_feture,
                                                    train_relative_positions,train_reverse_relative_positions,
                                                    train_eye_feature,train_eeg_feature,1,task_text)                  
                                    _, step, loss = sess.run([train_op,global_step,m.loss],feed_dict)
                                    if step % 100 == 0:
                                        temp = "step {},loss {}".format(step, loss/setting.batch_size)
                                        print(temp)
                                    if step % 500 == 0:
                                        saver.save(sess, save_path=save_path, global_step=step)
                                else:
                                    feed_dict = model_input(setting.batch_size,m,i,temp_order,train_word,
                                                    train_label,train_seqlen,train_pos_feture,
                                                    train_relative_positions,train_reverse_relative_positions,
                                                    train_eye_feature,train_eeg_feature,0,task_cognitive)
                                    _, step1, loss = sess.run([train_op1,global_step1,m.loss], feed_dict)
                                    if step1 % 100 == 0:
                                        temp = "step2 {},loss {}".format(step1, loss/setting.batch_size)
                                        print(temp)       
if __name__ == "__main__":
    tf.app.run()

