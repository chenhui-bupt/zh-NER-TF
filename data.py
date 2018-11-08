# -*- coding:utf-8 -*-
import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:  # 一句话
            data.append((sent_, tag_))  # 数据格式：list（句子（词序列），标签序列）
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)  # 读取中文语料数据
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():  # 所有数字都统一成'NUM'
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):  # 所有的英文字母都统一成'ENG'
                word = '<ENG>'
            if word not in word2id:  # word2id倒排索引和计数（id，count）
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':  # 统计低频词
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]  # 删除低频词

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id  # 重新编码
        new_id += 1
    word2id['<UNK>'] = new_id  # unknown
    word2id['<PAD>'] = 0  # pad是0，因为在对句子padding对齐的时候补的0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)  # 序列化存储


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """
    读取倒排索引和计数，{word:(id, count)}
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))  # 均匀初始化
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))  # 所有句子最长的长度
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)  # padding从后补零
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))  # 记录句子原始长度
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:  # 句子级shuffle，随机打乱句子
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)  # sentence的word id序列，转为数值型
        label_ = [tag2label[tag] for tag in tag_]  # 句子的label id序列，转为数值型

        if len(seqs) == batch_size:  # 当其够一个batch时，就yield出去
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:  # 最后不能整除的余数部分，也要yield
        yield seqs, labels

