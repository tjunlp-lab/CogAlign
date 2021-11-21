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
        self.clip=5 
        self.num_epoches=40
        self.pos_vocab_size=32
        self.relative_position_vocab_size=51
        self.reverse_position_vocab_size=53
        self.dim_position=20
        self.dim_pos=30
        self.num_task=2

class TransferModel(object):
    def __init__(self,setting,word_embed,is_adv,cog_signals,text_aware_att,tags_num,maxSentencelen,task):
        self.lr = setting.lr
        self.dim_word = setting.dim_word
        self.dim_lstm = setting.dim_lstm
        self.maxSentencelen = maxSentencelen
        self.keep_prob = setting.keep_prob
        self.batch_size = setting.batch_size
        self.pos_vocab_size = setting.pos_vocab_size
        self.relative_pos_vocab_size = setting.relative_position_vocab_size
        self.reverse_pos_vocab_size = setting.reverse_position_vocab_size
        self.dim_pos = setting.dim_pos
        self.dim_position = setting.dim_position
        self.word_embed = word_embed
        self.clip = setting.clip
        self.num_task = setting.num_task
        self.is_adv = is_adv
        self.cog_signals = cog_signals
        self.text_aware_att = text_aware_att
        self.tags_num = tags_num
        self.emb_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
        self.eye_input = tf.placeholder(tf.float32, [None, self.maxSentencelen,5])
        self.eeg_input = tf.placeholder(tf.float32, [None, self.maxSentencelen,105])
        self.label = tf.placeholder(tf.int32, [None, self.tags_num])
        self.task_label = tf.placeholder(tf.int32, [None,2])
        self.sentence_len = tf.placeholder(tf.int32, [None])
        self.is_text = tf.placeholder(dtype=tf.int32)
        self.task = task
        if self.task=='relation':
            self.pos_input = tf.placeholder(tf.int32, [None, self.maxSentencelen]) 
            self.relative_position_input = tf.placeholder(tf.int32, [None, self.maxSentencelen]) 
            self.reverse_position_input = tf.placeholder(tf.int32, [None, self.maxSentencelen])
            with tf.variable_scope("pos_feature"):
                self.W_pos = tf.Variable(tf.random_uniform([self.pos_vocab_size+1, self.dim_pos],-1,1),trainable=True, name="W_pos")
                self.pos_embedding_placeholder = tf.placeholder(tf.float32, [self.pos_vocab_size+1, self.dim_pos])
                pos_embedding_init = self.W_pos.assign(self.pos_embedding_placeholder)
            with tf.variable_scope("relative_position"):
                self.W_relative_positon = tf.Variable(tf.random_uniform([self.relative_pos_vocab_size+1, self.dim_position],-1,1),trainable=True, name="W_relative_position")
                self.relative_position_placeholder = tf.placeholder(tf.float32, [self.relative_pos_vocab_size+1, self.dim_position])
                relative_position_init = self.W_relative_positon.assign(self.relative_position_placeholder)
            with tf.variable_scope("reverse_position"):
                self.W_reverse_positon = tf.Variable(tf.random_uniform([self.reverse_pos_vocab_size+1, self.dim_position],-1,1),trainable=True, name="W_reverse_position")
                self.reverse_position_placeholder = tf.placeholder(tf.float32, [self.reverse_pos_vocab_size+1, self.dim_position])
                reverse_position_init = self.W_reverse_positon.assign(self.reverse_position_placeholder)
        with tf.variable_scope('word_embedding'):
            self.word_embedding = tf.get_variable(name='word_embedding', dtype=tf.float32,initializer=tf.cast(self.word_embed, tf.float32))
        with tf.variable_scope("eye_feature"):
            self.eye_embedding = tf.convert_to_tensor(self.eye_input)
              
        with tf.variable_scope("eeg_feature"):
            self.eeg_embedding = tf.convert_to_tensor(self.eeg_input)
            
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
        text_embedding = tf.nn.dropout(tf.nn.embedding_lookup(self.word_embedding, self.emb_input),self.keep_prob)
        eye_input = tf.cast(self.eye_embedding,dtype=tf.float32)
        eeg_input = tf.cast(self.eeg_embedding,dtype=tf.float32)
        if self.task=='re':
            pos_input = tf.nn.embedding_lookup(self.W_pos, self.pos_input)
            relative_position_input = tf.nn.embedding_lookup(self.W_relative_positon, self.relative_position_input)
            reverse_position_input = tf.nn.embedding_lookup(self.W_reverse_positon, self.reverse_position_input)
            text_embedding = tf.concat([text_embedding, pos_input, relative_position_input, reverse_position_input], axis=-1)
        if self.cog_signals=='eye_EEG':
            cog_embedding = tf.concat([eye_input, eeg_input], axis=-1)
        elif self.cog_signals=='eye':
            cog_embedding = eye_input
        elif self.cog_signals=='EEG':
            cog_embedding = eeg_input
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
            attention_output = self.self_attention(bilstm_output)
        dense_output = tf.nn.dropout(tf.layers.dense(inputs=attention_output,units=self.dim_lstm,activation=tf.nn.tanh),self.keep_prob)    
        self.text_project_logits = tf.layers.dense(inputs=dense_output,units=self.tags_num)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.text_project_logits, labels=self.label)) 
            
    def multi_task(self):
        text_embedding = tf.nn.dropout(tf.nn.embedding_lookup(self.word_embedding, self.emb_input),self.keep_prob)
        eye_input = tf.cast(self.eye_embedding,dtype=tf.float32)
        eeg_input = tf.cast(self.eeg_embedding,dtype=tf.float32)
        if self.task=='re':
            pos_input = tf.nn.embedding_lookup(self.W_pos, self.pos_input)
            relative_position_input = tf.nn.embedding_lookup(self.W_relative_positon, self.relative_position_input)
            reverse_position_input = tf.nn.embedding_lookup(self.W_reverse_positon, self.reverse_position_input)
            text_embedding = tf.concat([text_embedding, pos_input, relative_position_input, reverse_position_input], axis=-1)
        if self.cog_signals=='eye_EEG':
            cog_embedding = tf.concat([eye_input, eeg_input], axis=-1)
        elif self.cog_signals=='eye':
            cog_embedding = eye_input
        elif self.cog_signals=='EEG':
            cog_embedding = eeg_input
        if self.text_aware_att:
            cog_embedding = self.text_aware_attention(text_embedding,cog_embedding)
        text_hidden = text_embedding.get_shape().as_list()[-1]
        cog2text = tf.layers.dense(inputs=cog_embedding,units=text_hidden,activation=tf.nn.relu)
        share_input = tf.where(tf.equal(self.is_text,tf.ones_like(self.is_text)),text_embedding,cog2text)
    
        with tf.variable_scope('shared_bilstm'):
            shared_cell_fw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            shared_cell_bw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            shared_cell_fw = tf.nn.rnn_cell.DropoutWrapper(shared_cell_fw, output_keep_prob=self.keep_prob)
            shared_cell_bw = tf.nn.rnn_cell.DropoutWrapper(shared_cell_bw, output_keep_prob=self.keep_prob)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                shared_cell_fw, shared_cell_bw, share_input, sequence_length=self.sentence_len, dtype=tf.float32)
            self.shared_output = tf.concat([output_fw, output_bw], axis=-1)
            self.attention_output = self.self_attention(self.shared_output)

        with tf.variable_scope('cog_private_bilstm'):
            cog_private_cell_fw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            cog_private_cell_bw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            cog_private_cell_fw = tf.nn.rnn_cell.DropoutWrapper(cog_private_cell_fw, output_keep_prob=self.keep_prob)
            cog_private_cell_bw = tf.nn.rnn_cell.DropoutWrapper(cog_private_cell_bw, output_keep_prob=self.keep_prob)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cog_private_cell_fw, cog_private_cell_bw, cog_embedding, sequence_length=self.sentence_len, dtype=tf.float32)
            self.cog_private_output = tf.concat([output_fw, output_bw], axis=-1)
            cog_output = tf.concat([self.cog_private_output,self.shared_output],axis=-1)
            cog_output = self.self_attention(cog_output)

        with tf.variable_scope('text_private_bilstm'):
            text_private_cell_fw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            text_private_cell_bw = tf.contrib.rnn.LSTMCell(self.dim_lstm)
            text_private_cell_fw = tf.nn.rnn_cell.DropoutWrapper(text_private_cell_fw, output_keep_prob=self.keep_prob)
            text_private_cell_bw = tf.nn.rnn_cell.DropoutWrapper(text_private_cell_bw, output_keep_prob=self.keep_prob)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                text_private_cell_fw, text_private_cell_bw, text_embedding, sequence_length=self.sentence_len, dtype=tf.float32)
            self.text_private_output = tf.concat([output_fw, output_bw], axis=-1)
            text_output = tf.concat([self.text_private_output,self.shared_output],axis=-1)
            text_output = self.self_attention(text_output)

        text_output = tf.nn.dropout(tf.layers.dense(inputs=text_output,units=self.dim_lstm,activation=tf.nn.tanh),self.keep_prob)
        self.text_project_logits = tf.layers.dense(inputs=text_output,units=self.tags_num)
        with tf.variable_scope('text_loss'):
            self.text_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.text_project_logits, labels=self.label))

        cog_output = tf.nn.dropout(tf.layers.dense(inputs=cog_output,units=self.dim_lstm,activation=tf.nn.tanh),self.keep_prob)
        self.cog_project_logits = tf.layers.dense(inputs=cog_output,units=self.tags_num)
        with tf.variable_scope('cog_loss'):
            self.cog_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.cog_project_logits, labels=self.label))
        
        if self.is_adv == False:
            self.loss = tf.cast(self.is_text,tf.float32)*self.text_loss+tf.cast((1-self.is_text),tf.float32)*self.cog_loss
        elif self.is_adv == True:
            self.adv_loss = self.adversarial_loss(self.attention_output)
            self.loss = tf.cast(self.is_text,tf.float32)*self.text_loss+tf.cast((1-self.is_text),tf.float32)*self.cog_loss+self.adv_loss

