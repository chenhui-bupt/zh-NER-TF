# -*- coding:utf-8 -*-
import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from utils import get_logger
from eval import conlleval


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim  # lstm的hidden dim
        self.embeddings = embeddings  # 词向量（均匀初始化的）
        self.CRF = args.CRF  # 是否使用CRF层
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer  # 优化器
        self.lr = args.lr
        self.clip_grad = args.clip  # 梯度限制
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab  # 字典词表
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config  # tensorflow的配置

    def build_graph(self):  # 构建图的流程，前向运算
        self.add_placeholders()  # 数据输入
        self.lookup_layer_op()  # lookup层得到初始化的词向量
        self.biLSTM_layer_op()  # bilstm层
        self.softmax_pred_op()  # softmax计算所有标签的概率
        self.loss_op()  # 损失函数
        self.trainstep_op()  # 优化函数，（优化器，目标）
        self.init_op()  # tf变量初始化（局部，全局）

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")  # batchsize个句子，每个句子是word id序列
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")  # 标签，batch_size * 句子的标签序列
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")  # batch_size * 原始句子长度

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")  # dropout
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")  # 学习率

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,  # 是否可被更新，如果是pretrain的词向量可以选择'否'
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,  # 词向量
                                                     ids=self.word_ids,  # placeholder
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)  # dropout层

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)  # 前向lstm
            cell_bw = LSTMCell(self.hidden_dim)  # 后向lstm
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(  # 双向lstm
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)  # 双向lstm输出concat到一起，dim变成2倍
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):  # 对lstm输出做一层全连接层投影，投影到标签的类别数维度，可以直接做多分类任务，也可接CRF
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],  # 投影到标签数的维度
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),  # 0初始化
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])  # [(batch_size * sequence_length), 2hidden_dim]
            pred = tf.matmul(output, W) + b  # 全连接层,所以output要先变成二维，然后再变回去

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])  # s[1]是句子长度，[batch, sequence_length, num_tags]

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)  # sequence_length做mask用，计算损失函数
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)  # 多分类，最后一维上确定每一个单词的标签
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)  # global_step，不能trainable
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)  # 单独计算每一步梯度，为了做梯度限制
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)  # 更新梯度

    def init_op(self):
        self.init_op = tf.global_variables_initializer()  # 为什么么么有初始化局部变量

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()  # merge_all必须写在scalar等后面
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)  # 把图结构也存下来

    def train(self, train, dev):
        """

        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())  # 保存所有全局变量

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):  # 训练模型，并且每个epoch做一下验证
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):  # 建立会话session就能直接恢复模型
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)  # 恢复模型，model_path: 'model-31680'即可不用写后面的.meta或.data等
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def demo_one(self, sess, sent):
        """
        在线给一句话打标签
        :param sess:
        :param sent: 
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """
        run one epoch
        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size  # 计算batch的数量

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):

            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)  # feed dict喂数据
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:  # 每个epoch保存一次模型
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)  # 验证集验证
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)  # evaluate评估模型

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)  # 对句子进行padding，使所有句子长度对齐，补0

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)  # 标签labels也要补0对齐，0的标签是非实体(O)的意思
            feed_dict[self.labels] = labels_
        if lr is not None:  # 学习率
            feed_dict[self.lr_pl] = lr
        if dropout is not None:  # dropout
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list  # 喂数据feed dict，和句子原始长度

    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)  # 对一个batch进行预测
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list  # 所有的预测结果，用于后续的评价结果

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)  # 喂数据

        if self.CRF:  # 最后一层接了CRF层
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)  # 维特比译码
                label_list.append(viterbi_seq)
            return label_list, seq_len_list  # 维特比译码标签，句子长度

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)  # 无CRF层，直接多分类
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}  # tag_id2tag_name
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if len(label_) != len(sent):  # 如果预测的某句话的label与句子长度不一致
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):  # 记录这句话的[词，标签，模型预测标签]
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)  # 记录每句话的结果
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)

