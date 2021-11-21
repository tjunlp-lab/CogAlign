import tensorflow as tf
import numpy as np
import CogAlign
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os

def compute_tag(label):
    tag=[]
    for i in label:
        i=list(i)
        tag.append(i.index(max(i)))
    return tag

def cal_prf(true_label,pred_label,mode):
    acc = accuracy_score(true_label, pred_label)
    p = precision_score(true_label, pred_label, average=mode)
    r = recall_score(true_label, pred_label, average=mode)
    f1 = f1_score(true_label, pred_label, average=mode)
    return acc,p,r,f1

def main(_):
    data_path = ''
    embedding_path = ''
    model_path = ''
    is_adv = True
    is_attention = True
    cog_ = 'EEG'
    mode = 'multi'
    task = 'sentiment'
    fold_num = 10
    embedding = np.load(embedding_path)
    all_acc,all_p,all_r,all_f1 = 0,0,0,0
    setting = CogAlign.Setting()
    for fold in range(fold_num):
        print ('read test data......')
        test_word = np.load(data_path+'/fold_{}/test_word.npy'.format(fold))
        test_tag = np.load(data_path+'/fold_{}/test_tag.npy'.format(fold))
        test_length = np.load(data_path+'/fold_{}/test_seqlen.npy'.format(fold))
        test_eye_feature = np.load(data_path+'/fold_{}/test_eye_feature.npy'.format(fold))
        test_eeg_feature = np.load(data_path+'/fold_{}/test_eeg_feature.npy'.format(fold))
        sen_len = test_length[0]
        with tf.Graph().as_default():
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            with sess.as_default():
                with tf.variable_scope('sentiment_model'):
                    m = CogAlign.TransferModel(setting,tf.cast(embedding,tf.float32),is_adv=is_adv,cog_signals=cog_,
                                            text_aware_att=is_attention,tags_num=3,maxSentencelen=sen_len,task=task)
                if mode == 'single':
                    m.single_task()
                elif mode == 'multi':
                    m.multi_task()
                saver = tf.train.Saver()
                max_p,max_r,max_f = 0,0,0   
                for k in range(1000,6000,500):   
                    saver.restore(sess, model_path)
                    feed_dict = {}
                    feed_dict[m.emb_input] = test_word
                    feed_dict[m.sentence_len] = test_length
                    feed_dict[m.is_text] = 1
                    feed_dict[m.eye_input] = np.asarray(test_eye_feature)
                    feed_dict[m.eeg_input] = np.asarray(test_eeg_feature)
                    logits= sess.run([m.text_project_logits],feed_dict)
                    results=logits[0].tolist()
                    tag_pred = compute_tag(results)
                    current_acc,current_p,current_r,current_f=cal_prf(tag_pred,test_tag,'macro')
                    if current_f>max_f:
                        max_p = current_p
                        max_r = current_r
                        max_f = current_f
                print("Fold {} :test_P {},test_R {},test_F {}".format(fold, max_p, max_r, max_f))    
                all_p += max_p
                all_r += max_r
                all_f1 += max_f      
                print('avg_p {},avg_r {},avg_f {}'.format(all_p/fold_num, all_r/fold_num, all_f1/fold_num))

if __name__ == "__main__":
    tf.app.run()