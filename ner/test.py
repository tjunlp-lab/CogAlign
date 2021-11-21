import tensorflow as tf
import numpy as np
import CogAlign
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
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
    
def decode(logits,trans_params,lengths):
    viterbi_sequences=[]
    for logit, length in zip(logits, lengths):
        logit = logit[:length]
        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
        viterbi_sequences += [viterbi_seq]
    return viterbi_sequences
    
def get_ner_BIO(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '') & (index_tag != ''):
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix
    
def get_ner_fmeasure(golden_lists, predict_lists, label_type="BIO"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        gold_matrix = get_ner_BIO(golden_list)
        pred_matrix = get_ner_BIO(predict_list)
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (right_tag + 0.0) / all_tag
    return accuracy, precision, recall, f_measure
    
def evaluate(pred_value,label_value,length_value,word_value):
    id_tag={0:'O',1:'B-LOC',2:'I-LOC',3:'B-ORG',4:'I-ORG',5:'B-PER',6:'I-PER'}
    gold_label=[]
    pred_label=[]
    for i in range(len(pred_value)):
        result=[]
        temp_label=label_value[i][:length_value[i]]
        gold=[]
        pred=[]
        for j in range(length_value[i]):
            gold.append(id_tag[temp_label[j]])
            pred.append(id_tag[pred_value[i][j]])
        gold_label.append(gold)
        pred_label.append(pred)
    ACC,P,R,F=get_ner_fmeasure(gold_label,predict_label)
    return ACC,P,R,F

def main(_):
    data_path = ''
    embedding_path = ''
    model_path = ''
    is_adv = True
    is_attention = True
    cog_ = 'EEG'
    mode = 'multi'
    fold_num = 10
    embedding = np.load(embedding_path)
    char_vocab_size = 
    all_acc,all_p,all_r,all_f1 = 0,0,0,0
    setting = CogAlign.Setting()
    for fold in range(fold_num):
        print ('read test data......')
        test_word = np.load(data_path+'/fold_{}/test_word.npy'.format(fold))
        test_char = np.load(data_path+'/fold_{}/test_char.npy'.format(fold))
        test_tag = np.load(data_path+'/fold_{}/test_tag.npy'.format(fold))
        test_length = np.load(data_path+'/fold_{}/test_seqlen.npy'.format(fold))
        
        test_eye_tfd = np.load(data_path+'/fold_{}/test_eye_tfd.npy'.format(fold))
        test_eye_nf = np.load(data_path+'/fold_{}/test_eye_nf.npy'.format(fold))
        test_eye_ffd = np.load(data_path+'/fold_{}/test_eye_ffd.npy'.format(fold))
        test_eye_fpd = np.load(data_path+'/fold_{}/test_eye_fpd.npy'.format(fold))
        test_eye_fp = np.load(data_path+'/fold_{}/test_eye_fp.npy'.format(fold))
        test_eye_nr = np.load(data_path+'/fold_{}/test_eye_nr.npy'.format(fold))
        test_eye_rrp = np.load(data_path+'/fold_{}/test_eye_rrp.npy'.format(fold))
        test_eye_mfd = np.load(data_path+'/fold_{}/test_eye_mfd.npy'.format(fold))
        test_eye_trfd = np.load(data_path+'/fold_{}/test_eye_trfd.npy'.format(fold))
        test_eye_w2fp = np.load(data_path+'/fold_{}/test_eye_w2fp.npy'.format(fold))
        test_eye_w1fp = np.load(data_path+'/fold_{}/test_eye_w1fp.npy'.format(fold))
        test_eye_wp1fp = np.load(data_path+'/fold_{}/test_eye_wp1fp.npy'.format(fold))
        test_eye_wp2fp = np.load(data_path+'/fold_{}/test_eye_wp2fp.npy'.format(fold))
        test_eye_w2fd = np.load(data_path+'/fold_{}/test_eye_w2fd.npy'.format(fold))
        test_eye_w1fd = np.load(data_path+'/fold_{}/test_eye_w1fd.npy'.format(fold))
        test_eye_wp1fd = np.load(data_path+'/fold_{}/test_eye_wp1fd.npy'.format(fold))
        test_eye_wp2fd = np.load(data_path+'/fold_{}/test_eye_wp2fd.npy'.format(fold))
        
        test_eeg_t1 = np.load(data_path+'/fold_{}/test_eeg_t1.npy'.format(fold))
        test_eeg_t2 = np.load(data_path+'/fold_{}/test_eeg_t2.npy'.format(fold))
        test_eeg_a1 = np.load(data_path+'/fold_{}/test_eeg_a1.npy'.format(fold))
        test_eeg_a2 = np.load(data_path+'/fold_{}/test_eeg_a2.npy'.format(fold))
        test_eeg_b1 = np.load(data_path+'/fold_{}/test_eeg_b1.npy'.format(fold))
        test_eeg_b2 = np.load(data_path+'/fold_{}/test_eeg_b2.npy'.format(fold))
        test_eeg_g1 = np.load(data_path+'/fold_{}/test_eeg_g1.npy'.format(fold))
        test_eeg_g2 = np.load(data_path+'/fold_{}/test_eeg_g2.npy'.format(fold))
        seq_len = test_length[0]
        with tf.Graph().as_default():
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess=tf.Session(config=config)
            with sess.as_default():
                with tf.variable_scope('ner_model'):
                    m = CogAlign.TransferModel(setting,tf.cast(embedding,tf.float32),is_adv=is_adv,cog_signals=cog_,
                                                text_aware_att=is_attention,char_vocab_size=char_vocab_size,maxSentencelen=seq_len)
                    if mode == 'single':
                        m.single_task()
                    elif mode == 'multi':
                        m.multi_task()
                saver = tf.train.Saver()
                max_p,max_r,max_f = 0,0,0                       
                for k in range(1000,6000,500):   
                    saver.restore(sess, model_path)
                    feed_dict={}
                    feed_dict[m.emb_input] = np.asarray(test_word)
                    feed_dict[m.char_input] = np.asarray(test_char)
                    feed_dict[m.sentence_len] = np.asarray(test_length)
                    feed_dict[m.label] = np.asarray(test_tag)
                    feed_dict[m.eye_tfd_input] = np.asarray(test_eye_tfd)
                    feed_dict[m.eye_nf_input] = np.asarray(test_eye_nf)
                    feed_dict[m.eye_ffd_input] = np.asarray(test_eye_ffd)
                    feed_dict[m.eye_fpd_input] = np.asarray(test_eye_fpd)
                    feed_dict[m.eye_fp_input] = np.asarray(test_eye_fp)
                    feed_dict[m.eye_nr_input] = np.asarray(test_eye_nr)
                    feed_dict[m.eye_rrp_input] = np.asarray(test_eye_rrp)
                    feed_dict[m.eye_mfd_input] = np.asarray(test_eye_mfd)
                    feed_dict[m.eye_trfd_input] = np.asarray(test_eye_trfd)
                    feed_dict[m.eye_w2fp_input] = np.asarray(test_eye_w2fp)
                    feed_dict[m.eye_w1fp_input] = np.asarray(test_eye_w1fp)
                    feed_dict[m.eye_wp1fp_input] = np.asarray(test_eye_wp1fp)
                    feed_dict[m.eye_wp2fp_input] = np.asarray(test_eye_wp2fp)
                    feed_dict[m.eye_w2fd_input] = np.asarray(test_eye_w2fd)
                    feed_dict[m.eye_w1fd_input] = np.asarray(test_eye_w1fd)
                    feed_dict[m.eye_wp1fd_input] = np.asarray(test_eye_wp1fd)
                    feed_dict[m.eye_wp2fd_input] = np.asarray(test_eye_wp2fd)
                    feed_dict[m.eeg_t1_input] = np.asarray(test_eeg_t1)
                    feed_dict[m.eeg_t2_input] = np.asarray(test_eeg_t2)
                    feed_dict[m.eeg_a1_input] = np.asarray(test_eeg_a1)
                    feed_dict[m.eeg_a2_input] = np.asarray(test_eeg_a2)
                    feed_dict[m.eeg_b1_input] = np.asarray(test_eeg_b1)
                    feed_dict[m.eeg_b2_input] = np.asarray(test_eeg_b2)
                    feed_dict[m.eeg_g1_input] = np.asarray(test_eeg_g1)
                    feed_dict[m.eeg_g2_input] = np.asarray(test_eeg_g2)
                    feed_dict[m.is_text]=1
                    logits, trans_params = sess.run([m.text_project_logits,m.text_trans_params],feed_dict)
                    viterbi_sequences = decode(logits,trans_params,seq_len)
                    current_acc,current_p,current_r,current_f = evaluate(viterbi_sequences,test_tag,test_length,test_word)
                    if current_f > max_f:
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
   