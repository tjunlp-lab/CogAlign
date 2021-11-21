from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
import CogAlign
from tensorflow.python.framework import ops 
import os
import pickle

def savepickle(path,data):
    output=open(path,'wb')
    pickle.dump(data,output)
    output.close()
def loadpickle(name):
    pkl_file = open(path, 'rb')
    data1 = pickle.load(pkl_file)
    return data1
def model_input(batch,m,i,temp_order,train_word,train_char,train_tag,train_seqlen,train_eye_tfd,train_eye_nf,
               train_eye_ffd,train_eye_fpd,train_eye_fp,train_eye_nr,train_eye_rrp,train_eye_mfd,train_eye_trfd,
               train_eye_w2fp,train_eye_w1fp,train_eye_wp1fp,train_eye_wp2fp,train_eye_w2fd,train_eye_w1fd,
               train_eye_wp1fd,train_eye_wp2fd,train_eeg_t1,train_eeg_t2,train_eeg_a1,train_eeg_a2,
               train_eeg_b1,train_eeg_b2,train_eeg_g1,train_eeg_g2,is_text,task_label):
    temp_word = []
    temp_char = []
    temp_label = []
    temp_length = []
    temp_eye_tfd_feature,temp_eye_nf_feature,temp_eye_ffd_feature,temp_eye_fpd_feature = [],[],[],[]
    temp_eye_fp_feature,temp_eye_nr_feature,temp_eye_rrp_feature,temp_eye_mfd_feature = [],[],[],[]
    temp_eye_trfd_feature,temp_eye_w2fp_feature,temp_eye_w1fp_feature,temp_eye_wp1fp_feature = [],[],[],[]
    temp_eye_wp2fp_feature,temp_eye_w2fd_feature,temp_eye_w1fd_feature,temp_eye_wp1fd_feature = [],[],[],[]
    temp_eye_wp2fd_feature = []
    temp_eeg_t1_feature,temp_eeg_t2_feature,temp_eeg_a1_feature,temp_eeg_a2_feature = [],[],[],[]
    temp_eeg_b1_feature,temp_eeg_b2_feature,temp_eeg_g1_feature,temp_eeg_g2_feature = [],[],[],[]                                
    temp_input_index = temp_order[i * batch:(i + 1) * batch]
    for k in range(len(temp_input_index)):
        temp_word.append(train_word[temp_input_index[k]])
        temp_char.append(train_char[temp_input_index[k]])
        temp_label.append(train_tag[temp_input_index[k]])
        temp_length.append(train_seqlen[temp_input_index[k]])
        temp_eye_tfd_feature.append(train_eye_tfd[temp_input_index[k]])
        temp_eye_nf_feature.append(train_eye_nf[temp_input_index[k]])
        temp_eye_ffd_feature.append(train_eye_ffd[temp_input_index[k]])
        temp_eye_fpd_feature.append(train_eye_fpd[temp_input_index[k]])
        temp_eye_fp_feature.append(train_eye_fp[temp_input_index[k]])
        temp_eye_nr_feature.append(train_eye_nr[temp_input_index[k]])
        temp_eye_rrp_feature.append(train_eye_rrp[temp_input_index[k]])
        temp_eye_mfd_feature.append(train_eye_mfd[temp_input_index[k]])
        temp_eye_trfd_feature.append(train_eye_trfd[temp_input_index[k]])
        temp_eye_w2fp_feature.append(train_eye_w2fp[temp_input_index[k]])
        temp_eye_w1fp_feature.append(train_eye_w1fp[temp_input_index[k]])
        temp_eye_wp1fp_feature.append(train_eye_wp1fp[temp_input_index[k]])
        temp_eye_wp2fp_feature.append(train_eye_wp2fp[temp_input_index[k]])
        temp_eye_w2fd_feature.append(train_eye_w2fd[temp_input_index[k]])
        temp_eye_w1fd_feature.append(train_eye_w1fd[temp_input_index[k]])
        temp_eye_wp1fd_feature.append(train_eye_wp1fd[temp_input_index[k]])
        temp_eye_wp2fd_feature.append(train_eye_wp2fd[temp_input_index[k]])
        temp_eeg_t1_feature.append(train_eeg_t1[temp_input_index[k]])
        temp_eeg_t2_feature.append(train_eeg_t2[temp_input_index[k]])
        temp_eeg_a1_feature.append(train_eeg_a1[temp_input_index[k]])
        temp_eeg_a2_feature.append(train_eeg_a2[temp_input_index[k]])
        temp_eeg_b1_feature.append(train_eeg_b1[temp_input_index[k]])
        temp_eeg_b2_feature.append(train_eeg_b2[temp_input_index[k]])
        temp_eeg_g1_feature.append(train_eeg_g1[temp_input_index[k]])
        temp_eeg_g2_feature.append(train_eeg_g2[temp_input_index[k]])
    feed_dict = {}
    feed_dict[m.emb_input] = np.asarray(temp_word)
    feed_dict[m.char_input] = np.asarray(temp_char)
    feed_dict[m.label] = np.asarray(temp_label)
    feed_dict[m.sentence_len] = np.asarray(temp_length)
    feed_dict[m.eye_tfd_input] = np.asarray(temp_eye_tfd_feature)
    feed_dict[m.eye_nf_input] = np.asarray(temp_eye_nf_feature)
    feed_dict[m.eye_ffd_input] = np.asarray(temp_eye_ffd_feature)
    feed_dict[m.eye_fpd_input] = np.asarray(temp_eye_fpd_feature)
    feed_dict[m.eye_fp_input] = np.asarray(temp_eye_fp_feature)
    feed_dict[m.eye_nr_input] = np.asarray(temp_eye_nr_feature)
    feed_dict[m.eye_rrp_input] = np.asarray(temp_eye_rrp_feature)
    feed_dict[m.eye_mfd_input] = np.asarray(temp_eye_mfd_feature)
    feed_dict[m.eye_trfd_input] = np.asarray(temp_eye_trfd_feature)
    feed_dict[m.eye_w2fp_input] = np.asarray(temp_eye_w2fp_feature)
    feed_dict[m.eye_w1fp_input] = np.asarray(temp_eye_w1fp_feature)
    feed_dict[m.eye_wp1fp_input] = np.asarray(temp_eye_wp1fp_feature)
    feed_dict[m.eye_wp2fp_input] = np.asarray(temp_eye_wp2fp_feature)
    feed_dict[m.eye_w2fd_input] = np.asarray(temp_eye_w2fd_feature)
    feed_dict[m.eye_w1fd_input] = np.asarray(temp_eye_w1fd_feature)
    feed_dict[m.eye_wp1fd_input] = np.asarray(temp_eye_wp1fd_feature)
    feed_dict[m.eye_wp2fd_input] = np.asarray(temp_eye_wp2fd_feature)
    feed_dict[m.eeg_t1_input] = np.asarray(temp_eeg_t1_feature)
    feed_dict[m.eeg_t2_input] = np.asarray(temp_eeg_t2_feature)
    feed_dict[m.eeg_a1_input] = np.asarray(temp_eeg_a1_feature)
    feed_dict[m.eeg_a2_input] = np.asarray(temp_eeg_a2_feature)
    feed_dict[m.eeg_b1_input] = np.asarray(temp_eeg_b1_feature)
    feed_dict[m.eeg_b2_input] = np.asarray(temp_eeg_b2_feature)
    feed_dict[m.eeg_g1_input] = np.asarray(temp_eeg_g1_feature)
    feed_dict[m.eeg_g2_input] = np.asarray(temp_eeg_g2_feature)
    feed_dict[m.is_text] = is_text
    feed_dict[m.task_label] = np.asarray(task_label)
    return feed_dict
	
def main(_):
    data_path = ''
    embedding_path = ''
    model_path = ''
    is_adv = True
    is_attention = True
    cog_ = 'EEG'
    mode = 'multi'
    embedding = np.load(embedding_path)
    char_vocab_size = 
    fold_num = 10
    for fold in range(fold_num):
        setting = CogAlign.Setting()
        train_word = np.load(data_path+'/fold_{}/train_word.npy'.format(fold))
        train_char = np.load(data_path+'/fold_{}/train_char.npy'.format(fold))
        train_label = np.load(data_path+'/fold_{}/train_tag_sparse.npy'.format(fold))
        train_seqlen = np.load(data_path+'/fold_{}/train_seqlen.npy'.format(fold))
        train_eye_tfd = np.load(data_path+'/fold_{}/train_eye_tfd.npy'.format(fold))
        train_eye_nf = np.load(data_path+'/fold_{}/train_eye_nf.npy'.format(fold))
        train_eye_ffd = np.load(data_path+'/fold_{}/train_eye_ffd.npy'.format(fold))
        train_eye_fpd = np.load(data_path+'/fold_{}/train_eye_fpd.npy'.format(fold))
        train_eye_fp = np.load(data_path+'/fold_{}/train_eye_fp.npy'.format(fold))
        train_eye_nr = np.load(data_path+'/fold_{}/train_eye_nr.npy'.format(fold))
        train_eye_rrp = np.load(data_path+'/fold_{}/train_eye_rrp.npy'.format(fold))
        train_eye_mfd = np.load(data_path+'/fold_{}/train_eye_mfd.npy'.format(fold))
        train_eye_trfd = np.load(data_path+'/fold_{}/train_eye_trfd.npy'.format(fold))
        train_eye_w2fp = np.load(data_path+'/fold_{}/train_eye_w2fp.npy'.format(fold))
        train_eye_w1fp = np.load(data_path+'/fold_{}/train_eye_w1fp.npy'.format(fold))
        train_eye_wp1fp = np.load(data_path+'/fold_{}/train_eye_wp1fp.npy'.format(fold))
        train_eye_wp2fp = np.load(data_path+'/fold_{}/train_eye_wp2fp.npy'.format(fold))
        train_eye_w2fd = np.load(data_path+'/fold_{}/train_eye_w2fd.npy'.format(fold))
        train_eye_w1fd = np.load(data_path+'/fold_{}/train_eye_w1fd.npy'.format(fold))
        train_eye_wp1fd = np.load(data_path+'/fold_{}/train_eye_wp1fd.npy'.format(fold))
        train_eye_wp2fd = np.load(data_path+'/fold_{}/train_eye_wp2fd.npy'.format(fold))
        
        train_eeg_t1 = np.load(data_path+'/fold_{}/train_eeg_t1.npy'.format(fold))
        train_eeg_t2 = np.load(data_path+'/fold_{}/train_eeg_t2.npy'.format(fold))
        train_eeg_a1 = np.load(data_path+'/fold_{}/train_eeg_a1.npy'.format(fold))
        train_eeg_a2 = np.load(data_path+'/fold_{}/train_eeg_a2.npy'.format(fold))
        train_eeg_b1 = np.load(data_path+'/fold_{}/train_eeg_b1.npy'.format(fold))
        train_eeg_b2 = np.load(data_path+'/fold_{}/train_eeg_b2.npy'.format(fold))
        train_eeg_g1 = np.load(data_path+'/fold_{}/train_eeg_g1.npy'.format(fold))
        train_eeg_g2 = np.load(data_path+'/fold_{}/train_eeg_g2.npy'.format(fold))
        task_text = [[1,0]]*setting.batch_size
        task_cognitive = [[0,1]]*setting.batch_size 
        sen_len = train_seqlen[0] 
        save_path = model_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
                
        with tf.Graph().as_default():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            with sess.as_default():  
                initializer = tf.contrib.layers.xavier_initializer()
                with tf.variable_scope('ner_model',reuse=tf.AUTO_REUSE,initializer=initializer):
                    m = CogAlign.TransferModel(setting,tf.cast(embedding,tf.float32),is_adv=is_adv,cog_signals=cog_,
                                                text_aware_att=is_attention,char_vocab_size=char_vocab_size,maxSentencelen=seq_len)
                if mode == 'single':
                    m.single_task()
                elif mode == 'multi':
                    m.multi_task()
                global_step = tf.Variable(0, name="global_step", trainable=False)
                global_step1 = tf.Variable(0, name="global_step1", trainable=False)
                optimizer = tf.train.AdamOptimizer(0.001)
                if setting.clip>0:
                    grads, vs = zip(*optimizer.compute_gradients(m.loss))
                    grads,_  = tf.clip_by_global_norm(grads,clip_norm=setting.clip)
                    train_op = optimizer.apply_gradients(zip(grads,vs),global_step)
                    train_op1 = optimizer.apply_gradients(zip(grads, vs), global_step1)
                else:
                    train_op = optimizer.minimize(m.loss, global_step)
                    train_op1 = optimizer.minimize(m.loss, global_step1)

                sess.run(tf.global_variables_initializer())
                saver=tf.train.Saver(max_to_keep=None)
                
                for one_epoch in range(setting.num_epoches):
                    temp_order = list(range(len(train_word)))
                    np.random.shuffle(temp_order)
                    if mode == 'single':
                        for i in range(len(temp_order)//setting.batch_size):
                            feed_dict = model_input(setting.batch_size,m,i,temp_order,train_word,train_char,
                                        train_label,train_seqlen,train_eye_tfd,train_eye_nf,
                                        train_eye_ffd,train_eye_fpd,train_eye_fp,train_eye_nr,train_eye_rrp,train_eye_mfd,
                                        train_eye_trfd,train_eye_w2fp,train_eye_w1fp,train_eye_wp1fp,train_eye_wp2fp,train_eye_w2fd,
                                        train_eye_w1fd,train_eye_wp1fd,train_eye_wp2fd,train_eeg_t1,train_eeg_t2,train_eeg_a1,
                                        train_eeg_a2,train_eeg_b1,train_eeg_b2,train_eeg_g1,train_eeg_g2,1,task_text)                                
                            _, step, loss = sess.run([train_op,global_step,m.loss],feed_dict)
                            if step % 100 == 0:
                                temp = "step {},loss {}".format(step, loss/setting.batch_size)
                                print(temp)
                            if step % 500 == 0:
                                saver.save(sess, save_path=save_path, global_step=step)
                    elif mode == 'multi':
                        for i in range(len(temp_order)//setting.batch_size):
                            for j in range(2):
                                if j==0:
                                    feed_dict = model_input(setting.batch_size,m,i,temp_order,train_word,train_char,
                                        train_label,train_seqlen,train_eye_tfd,train_eye_nf,
                                        train_eye_ffd,train_eye_fpd,train_eye_fp,train_eye_nr,train_eye_rrp,train_eye_mfd,
                                        train_eye_trfd,train_eye_w2fp,train_eye_w1fp,train_eye_wp1fp,train_eye_wp2fp,train_eye_w2fd,
                                        train_eye_w1fd,train_eye_wp1fd,train_eye_wp2fd,train_eeg_t1,train_eeg_t2,train_eeg_a1,
                                        train_eeg_a2,train_eeg_b1,train_eeg_b2,train_eeg_g1,train_eeg_g2,1,task_text) 
                                    _, step, loss= sess.run([train_op,global_step,m.loss],feed_dict)
                                    if step % 100 == 0:
                                        temp = "step {},loss {}".format(step, loss/setting.batch_size)
                                        print(temp)
                                    if step % 500 == 0:
                                        saver.save(sess, save_path=save_path, global_step=step)
                                else:
                                    feed_dict = model_input(setting.batch_size,m,i,temp_order,train_word,train_char,
                                        train_label,train_seqlen,train_eye_tfd,train_eye_nf,
                                        train_eye_ffd,train_eye_fpd,train_eye_fp,train_eye_nr,train_eye_rrp,train_eye_mfd,
                                        train_eye_trfd,train_eye_w2fp,train_eye_w1fp,train_eye_wp1fp,train_eye_wp2fp,train_eye_w2fd,
                                        train_eye_w1fd,train_eye_wp1fd,train_eye_wp2fd,train_eeg_t1,train_eeg_t2,train_eeg_a1,
                                        train_eeg_a2,train_eeg_b1,train_eeg_b2,train_eeg_g1,train_eeg_g2,0,task_cognitive)  
                                    _, step1, loss= sess.run([train_op1,global_step1,m.loss], feed_dict)
                                    if step1 % 100 == 0:
                                        temp = "step2 {},loss {}".format(step1, loss/setting.batch_size)
                                        print(temp)       
if __name__ == "__main__":
    tf.app.run()

