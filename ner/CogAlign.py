import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.framework import ops

class FlipyGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0 
        
    def __call__(self, x, l=1.0):
        grad_name = "FlipyGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad): 
            return [tf.negative(grad)*l]

        g=tf.get_default_graph()
        with g.gradient_override_map({"Identity":grad_name}):
            y=tf.identity(x)

        self.num_calls+=1
        return y,grad_name  
class Setting(object):
    def __init__(self):
        self.lr=0.001
        self.dim_word=300 
        self.dim_lstm=50 
        self.keep_prob=0.7 
        self.batch_size=8
        self.num_task=2
        self.num_ner_tags=7 
        self.dim_char=30 
        self.dim_cognitive=1 
        self.maxWordlen=20 
        self.filter_size=3 
        self.num_filters=30 
        self.cog_vocab_size=8 
        self.num_epoches=40
        self.clip=5

class TransferModel(object):
    def __init__(self,setting,word_embed,is_adv,cog_signals,text_aware_att,char_vocab_size,maxSentencelen):
        self.lr = setting.lr
        self.dim_word = setting.dim_word
        self.dim_lstm = setting.dim_lstm
        self.maxSentencelen = maxSentencelen
        self.keep_prob = setting.keep_prob
        self.batch_size = setting.batch_size
        self.word_embed = word_embed
        self.cog_signals = cog_signals
        self.text_aware_att = text_aware_att
        self.cog_vocab_size = setting.cog_vocab_size
        self.char_vocab_size= char_vocab_size 
        self.dim_cognitive = setting.dim_cognitive
        self.num_task = setting.num_task
        self.is_adv = is_adv 
        self.num_ner_tags = setting.num_ner_tags
        self.dim_char=setting.dim_char 
        self.maxWordlen=setting.maxWordlen 
        self.filter_size=setting.filter_size 
        self.num_filters=setting.num_filters 
        
        self.emb_input = tf.placeholder(tf.int32, [None, self.maxSentencelen]) 
        self.char_input = tf.placeholder(tf.int32, [None, self.maxSentencelen*self.maxWordlen])
        self.eye_tfd_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_nf_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_ffd_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_fpd_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_fp_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_nr_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_rrp_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_mfd_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_trfd_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_w2fp_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_w1fp_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_wp1fp_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_wp2fp_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_w2fd_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_w1fd_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_wp1fd_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_wp2fd_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eeg_t1_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eeg_t2_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eeg_a1_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eeg_a2_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eeg_b1_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eeg_b2_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eeg_g1_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eeg_g2_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.label = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.task_label = tf.placeholder(tf.int32, [None,2])
        self.sentence_len = tf.placeholder(tf.int32, [None])
        self.is_text = tf.placeholder(dtype=tf.int32)

        with tf.variable_scope('word_embedding'): 
            self.word_embedding = tf.get_variable(name='word_embedding', dtype=tf.float32,initializer=tf.cast(self.word_embed, tf.float32))

        with tf.variable_scope("char_embedding"):
            self.W_char = tf.Variable(tf.random_uniform([self.char_vocab_size+1, self.dim_char],-1,1),trainable=True, name="W_char")
            self.char_embedding_placeholder = tf.placeholder(tf.float32, [self.char_vocab_size+1, self.dim_char])
            char_embedding_init = self.W_char.assign(self.char_embedding_placeholder)
            
        with tf.variable_scope("eye_tfd"):
            self.W_eye_tfd = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_tfd")
            self.eye_tfd_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_tfd_embedding_init = self.W_eye_tfd.assign(self.eye_tfd_embedding_placeholder)
            
        with tf.variable_scope("eye_nf"):
            self.W_eye_nf = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_nf")
            self.eye_nf_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_nf_embedding_init = self.W_eye_nf.assign(self.eye_nf_embedding_placeholder)

        with tf.variable_scope("eye_ffd"):
            self.W_eye_ffd = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_ffd")
            self.eye_ffd_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_ffd_embedding_init = self.W_eye_ffd.assign(self.eye_ffd_embedding_placeholder)

        with tf.variable_scope("eye_fpd"):
            self.W_eye_fpd = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_fpd")
            self.eye_fpd_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_fpd_embedding_init = self.W_eye_fpd.assign(self.eye_fpd_embedding_placeholder)

        with tf.variable_scope("eye_fp"):
            self.W_eye_fp = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_fp")
            self.eye_fp_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_fp_embedding_init = self.W_eye_fp.assign(self.eye_fp_embedding_placeholder)

        with tf.variable_scope("eye_nr"):
            self.W_eye_nr = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_nr")
            self.eye_nr_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_nr_embedding_init = self.W_eye_nr.assign(self.eye_nr_embedding_placeholder)

        with tf.variable_scope("eye_rrp"):
            self.W_eye_rrp = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_rrp")
            self.eye_rrp_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_rrp_embedding_init = self.W_eye_rrp.assign(self.eye_rrp_embedding_placeholder)

        with tf.variable_scope("eye_mfd"):
            self.W_eye_mfd = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_mfd")
            self.eye_mfd_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_mfd_embedding_init = self.W_eye_mfd.assign(self.eye_mfd_embedding_placeholder)

        with tf.variable_scope("eye_trfd"):
            self.W_eye_trfd = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_trfd")
            self.eye_trfd_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_trfd_embedding_init = self.W_eye_trfd.assign(self.eye_trfd_embedding_placeholder)

        with tf.variable_scope("eye_w2fp"):
            self.W_eye_w2fp = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_w2fp")
            self.eye_w2fp_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_w2fp_embedding_init = self.W_eye_w2fp.assign(self.eye_w2fp_embedding_placeholder)

        with tf.variable_scope("eye_w1fp"):
            self.W_eye_w1fp = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_w1fp")
            self.eye_w1fp_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_w1fp_embedding_init = self.W_eye_w1fp.assign(self.eye_w1fp_embedding_placeholder)

        with tf.variable_scope("eye_wp1fp"):
            self.W_eye_wp1fp = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_wp1fp")
            self.eye_wp1fp_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_wp1fp_embedding_init = self.W_eye_wp1fp.assign(self.eye_wp1fp_embedding_placeholder)

        with tf.variable_scope("eye_wp2fp"):
            self.W_eye_wp2fp = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_wp2fp")
            self.eye_wp2fp_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_wp2fp_embedding_init = self.W_eye_wp2fp.assign(self.eye_wp2fp_embedding_placeholder)

        with tf.variable_scope("eye_w2fd"):
            self.W_eye_w2fd = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_w2fd")
            self.eye_w2fd_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_w2fd_embedding_init = self.W_eye_w2fd.assign(self.eye_w2fd_embedding_placeholder)

        with tf.variable_scope("eye_w1fd"):
            self.W_eye_w1fd = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_w1fd")
            self.eye_w1fd_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_w1fd_embedding_init = self.W_eye_w1fd.assign(self.eye_w1fd_embedding_placeholder)

        with tf.variable_scope("eye_wp1fd"):
            self.W_eye_wp1fd = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_wp1fd")
            self.eye_wp1fd_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_wp1fd_embedding_init = self.W_eye_wp1fd.assign(self.eye_wp1fd_embedding_placeholder)

        with tf.variable_scope("eye_wp2fd"):
            self.W_eye_wp2fd = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eye_wp2fd")
            self.eye_wp2fd_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eye_wp2fd_embedding_init = self.W_eye_wp2fd.assign(self.eye_wp2fd_embedding_placeholder)

        with tf.variable_scope("eeg_t1"):
            self.W_eeg_t1 = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eeg_t1")
            self.eeg_t1_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eeg_t1_embedding_init = self.W_eeg_t1.assign(self.eeg_t1_embedding_placeholder)

            
        with tf.variable_scope("eeg_t2"):
            self.W_eeg_t2 = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eeg_t2")
            self.eeg_t2_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eeg_t2_embedding_init = self.W_eeg_t2.assign(self.eeg_t2_embedding_placeholder)

        with tf.variable_scope("eeg_a1"):
            self.W_eeg_a1 = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eeg_a1")
            self.eeg_a1_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eeg_a1_embedding_init = self.W_eeg_a1.assign(self.eeg_a1_embedding_placeholder)

        with tf.variable_scope("eeg_a2"):
            self.W_eeg_a2 = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eeg_a2")
            self.eeg_a2_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eeg_a2_embedding_init = self.W_eeg_a2.assign(self.eeg_a2_embedding_placeholder)

        with tf.variable_scope("eeg_b1"):
            self.W_eeg_b1 = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eeg_b1")
            self.eeg_b1_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eeg_b1_embedding_init = self.W_eeg_b1.assign(self.eeg_b1_embedding_placeholder)

        with tf.variable_scope("eeg_b2"):
            self.W_eeg_b2 = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eeg_b2")
            self.eeg_b2_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eeg_b2_embedding_init = self.W_eeg_b2.assign(self.eeg_b2_embedding_placeholder)

        with tf.variable_scope("eeg_g1"):
            self.W_eeg_g1 = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eeg_g1")
            self.eeg_g1_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eeg_g1_embedding_init = self.W_eeg_g1.assign(self.eeg_g1_embedding_placeholder)

        with tf.variable_scope("eeg_g2"):
            self.W_eeg_g2 = tf.Variable(tf.random_uniform([self.cog_vocab_size+1, self.dim_cognitive],-1,1),trainable=True, name="W_eeg_g2")
            self.eeg_g2_embedding_placeholder = tf.placeholder(tf.float32, [self.cog_vocab_size+1, self.dim_cognitive])
            eeg_g2_embedding_init = self.W_eeg_g2.assign(self.eeg_g2_embedding_placeholder)
      
    def adversarial_loss(self,feature):
        flip_gradient = FlipyGradientBuilder()
        feature,grab_name=flip_gradient(feature)
        feature=tf.nn.dropout(feature,self.keep_prob)
        logits = tf.layers.dense(inputs=feature,units=self.num_task,activation=tf.nn.relu)
        adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.task_label))
        return adv_loss

    def text_aware_attention(self,text,cog, scope='text_aware_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            d_in = text.get_shape().as_list()[1]
            d_out = cog.get_shape().as_list()[1]
            hidden_state = cog.get_shape().as_list()[-1]
            b = tf.Variable(tf.constant(0.1, shape=[hidden_state]), name="b")
            U = tf.get_variable(name='U', shape=[d_in, d_out], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            text_T = tf.matmul(tf.transpose(text, [0, 2, 1]),U)
            G = tf.nn.tanh(tf.matmul(text_T,cog)+b)
            score = tf.reduce_max(G, axis=1, keepdims=True)
            score = tf.nn.softmax(score)
            outputs = tf.multiply(score, cog)
        return outputs
    
    def self_attention(self,features):
        with tf.variable_scope("self_attention_layer"):
            hidden_state = features.get_shape().as_list()[-1]
            features = tf.layers.dense(features, hidden_state, activation=tf.nn.tanh)
            v = tf.Variable(tf.truncated_normal([hidden_state,1]), name='v')
            score = tf.matmul(features, v)
            score = tf.nn.softmax(score,dim=1)
            attention = tf.multiply(features,score)
            outputs = tf.reduce_sum(attention, axis=-2)
        return outputs
    
    def single_task(self):
        text_input = tf.nn.dropout(tf.nn.embedding_lookup(self.word_embedding, self.emb_input, name="word_embeddings"),self.keep_prob)
        char_input = tf.nn.dropout(tf.nn.embedding_lookup(self.W_char, self.char_input,name="char_embeddings"),self.keep_prob)
        eye_tfd_input = tf.nn.embedding_lookup(self.W_eye_tfd, self.eye_tfd_input,name="eye_tfd_embeddings")
        eye_nf_input = tf.nn.embedding_lookup(self.W_eye_nf, self.eye_nf_input,name="eye_nf_embeddings")
        eye_ffd_input = tf.nn.embedding_lookup(self.W_eye_ffd, self.eye_ffd_input,name="eye_ffd_embeddings")
        eye_fpd_input = tf.nn.embedding_lookup(self.W_eye_fpd, self.eye_fpd_input,name="eye_fpd_embeddings")
        eye_fp_input = tf.nn.embedding_lookup(self.W_eye_fp, self.eye_fp_input,name="eye_fp_embeddings")
        eye_nr_input = tf.nn.embedding_lookup(self.W_eye_nr, self.eye_nr_input,name="eye_nr_embeddings")
        eye_rrp_input = tf.nn.embedding_lookup(self.W_eye_rrp, self.eye_rrp_input,name="eye_rrp_embeddings")
        eye_mfd_input = tf.nn.embedding_lookup(self.W_eye_mfd, self.eye_mfd_input,name="eye_mfd_embeddings")
        eye_trfd_input = tf.nn.embedding_lookup(self.W_eye_trfd, self.eye_trfd_input,name="eye_trfd_embeddings")
        eye_w2fp_input = tf.nn.embedding_lookup(self.W_eye_w2fp, self.eye_w2fp_input,name="eye_w2fp_embeddings")
        eye_w1fp_input = tf.nn.embedding_lookup(self.W_eye_w1fp, self.eye_w1fp_input,name="eye_w1fp_embeddings")
        eye_wp1fp_input = tf.nn.embedding_lookup(self.W_eye_wp1fp, self.eye_wp1fp_input,name="eye_wp1fp_embeddings")
        eye_wp2fp_input = tf.nn.embedding_lookup(self.W_eye_wp2fp, self.eye_wp2fp_input,name="eye_wp2fp_embeddings")
        eye_w2fd_input = tf.nn.embedding_lookup(self.W_eye_w2fd, self.eye_w2fd_input,name="eye_w2fd_embeddings")
        eye_w1fd_input = tf.nn.embedding_lookup(self.W_eye_w1fd, self.eye_w1fd_input,name="eye_w1fd_embeddings")
        eye_wp1fd_input = tf.nn.embedding_lookup(self.W_eye_wp1fd, self.eye_wp1fd_input,name="eye_wp1fd_embeddings")
        eye_wp2fd_input = tf.nn.embedding_lookup(self.W_eye_wp2fd, self.eye_wp2fd_input,name="eye_wp2fd_embeddings")
        eeg_t1_input = tf.nn.embedding_lookup(self.W_eeg_t1, self.eeg_t1_input,name="eeg_t1_embeddings")
        eeg_t2_input = tf.nn.embedding_lookup(self.W_eeg_t2, self.eeg_t2_input,name="eeg_t2_embeddings")
        eeg_a1_input = tf.nn.embedding_lookup(self.W_eeg_a1, self.eeg_a1_input,name="eeg_a1_embeddings")
        eeg_a2_input = tf.nn.embedding_lookup(self.W_eeg_a2, self.eeg_a2_input,name="eeg_a2_embeddings")
        eeg_b1_input = tf.nn.embedding_lookup(self.W_eeg_b1, self.eeg_b1_input,name="eeg_b1_embeddings")
        eeg_b2_input = tf.nn.embedding_lookup(self.W_eeg_b2, self.eeg_b2_input,name="eeg_b2_embeddings")
        eeg_g1_input = tf.nn.embedding_lookup(self.W_eeg_g1, self.eeg_g1_input,name="eeg_g1_embeddings")
        eeg_g2_input = tf.nn.embedding_lookup(self.W_eeg_g2, self.eeg_g2_input,name="eeg_g2_embeddings")
        if self.cog_signals=='eye_EEG':
            cog_embedding = tf.concat([eye_tfd_input, eye_nf_input,eye_ffd_input,eye_fpd_input,eye_fp_input,eye_nr_input,eye_rrp_input,
                                eye_mfd_input,eye_trfd_input,eye_w2fp_input,eye_w1fp_input,eye_wp1fp_input,eye_wp2fp_input,
                                eye_w2fd_input,eye_w1fd_input,eye_wp1fd_input,eye_wp2fd_input,
                                eeg_t1_input,eeg_t2_input,eeg_a1_input,eeg_a2_input,eeg_b1_input,eeg_b2_input,eeg_g1_input,eeg_g2_input], axis=-1)
        elif self.cog_signals=='eye':
            cog_embedding = tf.concat([eye_tfd_input, eye_nf_input,eye_ffd_input,eye_fpd_input,eye_fp_input,eye_nr_input,eye_rrp_input,
                                eye_mfd_input,eye_trfd_input,eye_w2fp_input,eye_w1fp_input,eye_wp1fp_input,eye_wp2fp_input,
                                eye_w2fd_input,eye_w1fd_input,eye_wp1fd_input,eye_wp2fd_input],axis=-1)
        elif self.cog_signals=='EEG':
            cog_embedding = tf.concat([eeg_t1_input,eeg_t2_input,eeg_a1_input,eeg_a2_input,
                                         eeg_b1_input,eeg_b2_input,eeg_g1_input,eeg_g2_input],axis=-1)
        with tf.variable_scope('charCNN'):
            filter_shape = [self.filter_size, self.dim_char,self.num_filters]
            W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_conv")
            b_conv = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b_conv")
            conv = tf.nn.conv1d(char_input,
                            W_conv,
                            stride=1,
                            padding="SAME",
                            name="conv")
           
            expand = tf.expand_dims(conv, -1)
            pooled = tf.nn.max_pool(
                            expand,
                            ksize=[1,self.maxSentencelen * self.maxWordlen,1, 1], 
                            strides=[1, self.maxWordlen, 1, 1],
                            padding='SAME',
                            name="max_pool")
            self.char_pool_flat = tf.reshape(pooled, [-1,self.maxSentencelen,self.num_filters],name="char_pool_flat")
            text_embedding = tf.concat([text_input, self.char_pool_flat], axis=-1)
            text_embedding = tf.nn.dropout(text_embedding,self.keep_prob)
        if self.text_aware_att:
            cog_embedding = self.text_aware_attention(text_embedding,cog_embedding)
        concat_embedding = tf.concat([text_embedding, cog_embedding], axis=-1)
        with tf.variable_scope('bilstm'):
            private_cell_fw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            private_cell_bw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            private_cell_fw = tf.nn.rnn_cell.DropoutWrapper(private_cell_fw, output_keep_prob=self.keep_prob)
            private_cell_bw = tf.nn.rnn_cell.DropoutWrapper(private_cell_bw, output_keep_prob=self.keep_prob)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                private_cell_fw, private_cell_bw, concat_embedding, sequence_length=self.sentence_len, dtype=tf.float32)
            bilstm_output = tf.concat([output_fw, output_bw], axis=-1)
        text_pred = tf.nn.dropout(tf.layers.dense(bilstm_output, self.num_ner_tags, activation=tf.nn.relu),self.keep_prob)
        self.text_project_logits = tf.reshape(text_pred, [-1, self.maxSentencelen, self.num_ner_tags])
        with tf.variable_scope('loss'):
            log_likelihood, self.text_trans_params = tf.contrib.crf.crf_log_likelihood(inputs=self.text_project_logits,
                                          tag_indices=self.label,sequence_lengths=self.sentence_len)
        self.loss = tf.reduce_mean(-log_likelihood)   
    def multi_task(self):
        text_input = tf.nn.dropout(tf.nn.embedding_lookup(self.word_embedding, self.emb_input, name="word_embeddings"),self.keep_prob)
        char_input = tf.nn.dropout(tf.nn.embedding_lookup(self.W_char, self.char_input,name="char_embeddings"),self.keep_prob)
        eye_tfd_input = tf.nn.embedding_lookup(self.W_eye_tfd, self.eye_tfd_input,name="eye_tfd_embeddings")
        eye_nf_input = tf.nn.embedding_lookup(self.W_eye_nf, self.eye_nf_input,name="eye_nf_embeddings")
        eye_ffd_input = tf.nn.embedding_lookup(self.W_eye_ffd, self.eye_ffd_input,name="eye_ffd_embeddings")
        eye_fpd_input = tf.nn.embedding_lookup(self.W_eye_fpd, self.eye_fpd_input,name="eye_fpd_embeddings")
        eye_fp_input = tf.nn.embedding_lookup(self.W_eye_fp, self.eye_fp_input,name="eye_fp_embeddings")
        eye_nr_input = tf.nn.embedding_lookup(self.W_eye_nr, self.eye_nr_input,name="eye_nr_embeddings")
        eye_rrp_input = tf.nn.embedding_lookup(self.W_eye_rrp, self.eye_rrp_input,name="eye_rrp_embeddings")
        eye_mfd_input = tf.nn.embedding_lookup(self.W_eye_mfd, self.eye_mfd_input,name="eye_mfd_embeddings")
        eye_trfd_input = tf.nn.embedding_lookup(self.W_eye_trfd, self.eye_trfd_input,name="eye_trfd_embeddings")
        eye_w2fp_input = tf.nn.embedding_lookup(self.W_eye_w2fp, self.eye_w2fp_input,name="eye_w2fp_embeddings")
        eye_w1fp_input = tf.nn.embedding_lookup(self.W_eye_w1fp, self.eye_w1fp_input,name="eye_w1fp_embeddings")
        eye_wp1fp_input = tf.nn.embedding_lookup(self.W_eye_wp1fp, self.eye_wp1fp_input,name="eye_wp1fp_embeddings")
        eye_wp2fp_input = tf.nn.embedding_lookup(self.W_eye_wp2fp, self.eye_wp2fp_input,name="eye_wp2fp_embeddings")
        eye_w2fd_input = tf.nn.embedding_lookup(self.W_eye_w2fd, self.eye_w2fd_input,name="eye_w2fd_embeddings")
        eye_w1fd_input = tf.nn.embedding_lookup(self.W_eye_w1fd, self.eye_w1fd_input,name="eye_w1fd_embeddings")
        eye_wp1fd_input = tf.nn.embedding_lookup(self.W_eye_wp1fd, self.eye_wp1fd_input,name="eye_wp1fd_embeddings")
        eye_wp2fd_input = tf.nn.embedding_lookup(self.W_eye_wp2fd, self.eye_wp2fd_input,name="eye_wp2fd_embeddings")
        eeg_t1_input = tf.nn.embedding_lookup(self.W_eeg_t1, self.eeg_t1_input,name="eeg_t1_embeddings")
        eeg_t2_input = tf.nn.embedding_lookup(self.W_eeg_t2, self.eeg_t2_input,name="eeg_t2_embeddings")
        eeg_a1_input = tf.nn.embedding_lookup(self.W_eeg_a1, self.eeg_a1_input,name="eeg_a1_embeddings")
        eeg_a2_input = tf.nn.embedding_lookup(self.W_eeg_a2, self.eeg_a2_input,name="eeg_a2_embeddings")
        eeg_b1_input = tf.nn.embedding_lookup(self.W_eeg_b1, self.eeg_b1_input,name="eeg_b1_embeddings")
        eeg_b2_input = tf.nn.embedding_lookup(self.W_eeg_b2, self.eeg_b2_input,name="eeg_b2_embeddings")
        eeg_g1_input = tf.nn.embedding_lookup(self.W_eeg_g1, self.eeg_g1_input,name="eeg_g1_embeddings")
        eeg_g2_input = tf.nn.embedding_lookup(self.W_eeg_g2, self.eeg_g2_input,name="eeg_g2_embeddings")
        if self.cog_signals=='eye_EEG':
            cog_embedding = tf.concat([eye_tfd_input, eye_nf_input,eye_ffd_input,eye_fpd_input,eye_fp_input,eye_nr_input,eye_rrp_input,
                                eye_mfd_input,eye_trfd_input,eye_w2fp_input,eye_w1fp_input,eye_wp1fp_input,eye_wp2fp_input,
                                eye_w2fd_input,eye_w1fd_input,eye_wp1fd_input,eye_wp2fd_input,
                                eeg_t1_input,eeg_t2_input,eeg_a1_input,eeg_a2_input,eeg_b1_input,eeg_b2_input,eeg_g1_input,eeg_g2_input], axis=-1)
        elif self.cog_signals=='eye':
            cog_embedding = tf.concat([eye_tfd_input, eye_nf_input,eye_ffd_input,eye_fpd_input,eye_fp_input,eye_nr_input,eye_rrp_input,
                                eye_mfd_input,eye_trfd_input,eye_w2fp_input,eye_w1fp_input,eye_wp1fp_input,eye_wp2fp_input,
                                eye_w2fd_input,eye_w1fd_input,eye_wp1fd_input,eye_wp2fd_input],axis=-1)
        elif self.cog_signals=='EEG':
            cog_embedding = tf.concat([eeg_t1_input,eeg_t2_input,eeg_a1_input,eeg_a2_input,
                                         eeg_b1_input,eeg_b2_input,eeg_g1_input,eeg_g2_input],axis=-1)
        with tf.variable_scope('charCNN'):
            filter_shape = [self.filter_size, self.dim_char,self.num_filters]
            W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_conv")
            b_conv = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b_conv")
            conv = tf.nn.conv1d(char_input,
                            W_conv,
                            stride=1,
                            padding="SAME",
                            name="conv")
           
            expand = tf.expand_dims(conv, -1)
            pooled = tf.nn.max_pool(
                            expand,
                            ksize=[1,self.maxSentencelen * self.maxWordlen,1, 1], 
                            strides=[1, self.maxWordlen, 1, 1],
                            padding='SAME',
                            name="max_pool")
            self.char_pool_flat = tf.reshape(pooled, [-1,self.maxSentencelen,self.num_filters],name="char_pool_flat")
            text_embedding = tf.concat([text_input, self.char_pool_flat], axis=-1)
            text_embedding = tf.nn.dropout(text_embedding,self.keep_prob)
        if self.text_aware_att:
            cog_embedding = self.text_aware_attention(text_embedding,cog_embedding)
        text_hidden = text_embedding.get_shape().as_list()[-1]
        cog2text = tf.layers.dense(inputs=cog_embedding,units=text_hidden,activation=tf.nn.relu)
        share_input = tf.where(tf.equal(self.is_text,tf.ones_like(self.is_text)),text_embedding,cog2text)
        
        with tf.variable_scope('shared_bilstm'):
            shared_cell_fw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            shared_cell_bw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            shared_cell_fw = tf.nn.rnn_cell.DropoutWrapper(shared_cell_fw,output_keep_prob=self.keep_prob)
            shared_cell_bw = tf.nn.rnn_cell.DropoutWrapper(shared_cell_bw,output_keep_prob=self.keep_prob)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                shared_cell_fw, shared_cell_bw, share_input, sequence_length=self.sentence_len, dtype=tf.float32)
            self.shared_output = tf.concat([output_fw, output_bw], axis=-1)
            self.attention_output = self.self_attention(self.shared_output)

        with tf.variable_scope('cog_private_bilstm'):
            cog_private_cell_fw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            cog_private_cell_bw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            cog_private_cell_fw = tf.nn.rnn_cell.DropoutWrapper(cog_private_cell_fw,output_keep_prob=self.keep_prob)
            cog_private_cell_bw = tf.nn.rnn_cell.DropoutWrapper(cog_private_cell_bw,output_keep_prob=self.keep_prob)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cog_private_cell_fw, cog_private_cell_bw, cog_embedding, sequence_length=self.sentence_len, dtype=tf.float32)
            self.cog_private_output = tf.concat([output_fw, output_bw], axis=-1)
#             cog_private_output=self.self_attention(cog_private_output)

        with tf.variable_scope('text_private_bilstm'):
            text_private_cell_fw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            text_private_cell_bw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            text_private_cell_fw = tf.nn.rnn_cell.DropoutWrapper(text_private_cell_fw, output_keep_prob=self.keep_prob)
            text_private_cell_bw = tf.nn.rnn_cell.DropoutWrapper(text_private_cell_bw, output_keep_prob=self.keep_prob)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                text_private_cell_fw, text_private_cell_bw, text_embedding, sequence_length=self.sentence_len, dtype=tf.float32)
            self.text_private_output = tf.concat([output_fw, output_bw], axis=-1)
#             text_private_output = self.self_attention(self.text_private_output)

        text_output = tf.concat([self.text_private_output,self.shared_output],axis=-1)
        text_output = tf.nn.dropout(tf.layers.dense(text_output, self.dim_lstm, activation=tf.nn.tanh),self.keep_prob)
        text_pred = tf.nn.dropout(tf.layers.dense(text_output, self.num_ner_tags, activation=tf.nn.relu),self.keep_prob)
        self.text_project_logits = tf.reshape(text_pred, [-1, self.maxSentencelen, self.num_ner_tags])
        with tf.variable_scope('text_loss'):
            log_likelihood, self.text_trans_params = tf.contrib.crf.crf_log_likelihood(inputs=self.text_project_logits,
                                          tag_indices=self.label,sequence_lengths=self.sentence_len)
        self.text_loss = tf.reduce_mean(-log_likelihood) 

        cog_output = tf.concat([self.cog_private_output, self.shared_output],axis=-1)
        cog_output = tf.nn.dropout(tf.layers.dense(cog_output, self.dim_lstm, activation=tf.nn.tanh),self.keep_prob)
        cog_pred = tf.nn.dropout(tf.layers.dense(cog_output, self.num_ner_tags, activation=tf.nn.relu),self.keep_prob)
        self.cog_project_logits = tf.reshape(cog_pred, [-1, self.maxSentencelen, self.num_ner_tags])
        with tf.variable_scope('cog_loss'):
            log_likelihood, self.cog_trans_params = tf.contrib.crf.crf_log_likelihood(inputs=self.cog_project_logits,
                                                              tag_indices=self.label,
                                                              sequence_lengths=self.sentence_len)
        self.cog_loss = tf.reduce_mean(-log_likelihood)

        if self.is_adv == False:
            self.loss = tf.cast(self.is_text,tf.float32)*self.text_loss+tf.cast((1-self.is_text),tf.float32)*self.cog_loss
        elif self.is_adv == True:
            self.adv_loss = self.adversarial_loss(self.attention_output)
            self.loss = tf.cast(self.is_text,tf.float32)*self.text_loss+tf.cast((1-self.is_text),tf.float32)*self.cog_loss+self.adv_loss

