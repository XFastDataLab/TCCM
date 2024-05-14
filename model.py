#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import math
import random

import numpy as np
# import keras.layers
from AttentionModel import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import tensorflow_ranking as tfr


def BPR_LOSS(y_true, y_pred):
    x = tf.where(tf.equal(y_true, 1), y_pred, y_pred)
    x1, x2 = tf.split(x, 2, 1)
    bpr_loss = -K.mean(tf.math.log(tf.sigmoid(tf.subtract(x1, x2))))
    return bpr_loss


def ELU_activation(score):
    x1 = keras.layers.Lambda(lambda x: tf.math.exp(x))(score)
    x2 = keras.layers.Lambda(lambda x: x + 1)(score)
    return tf.where(score <= 0, x1, x2)


def get_embedding_encoder(config, entity_embedding_layer):
    LengthTable = {'title': config['title_length'],
                   'entity': config['max_entity_num']}
    input_length = 0
    PositionTable = {}
    for v in config['attrs']:
        PositionTable[v] = (input_length, input_length + LengthTable[v])
        input_length += LengthTable[v]
    print(PositionTable)
    news_input = Input((input_length,), dtype='int32')
    entity_input = keras.layers.Lambda(lambda x: x[:, PositionTable['entity'][0]:PositionTable['entity'][1]])(
        news_input)
    entity_emb = entity_embedding_layer(entity_input)
    model = Model(news_input, entity_emb)
    return model


def get_news_encoder_co1(config, word_num, word_embedding_matrix, entity_embedding_layer, seed):
    LengthTable = {'title': config['title_length'],
                   'entity': config['max_entity_num']}
    input_length = 0
    PositionTable = {}
    for v in config['attrs']:
        PositionTable[v] = (input_length, input_length + LengthTable[v])
        input_length += LengthTable[v]
    print(PositionTable)
    word_embedding_layer = Embedding(word_num + 1, word_embedding_matrix.shape[1], weights=[word_embedding_matrix],
                                     trainable=True)
    news_input = Input((input_length,), dtype='int32')

    title_input = keras.layers.Lambda(lambda x: x[:, PositionTable['title'][0]:PositionTable['title'][1]])(news_input)
    entity_input = keras.layers.Lambda(lambda x: x[:, PositionTable['entity'][0]:PositionTable['entity'][1]])(
        news_input)

    title_emb = word_embedding_layer(title_input)
    # title_emb = Dropout(0.2)(title_emb)

    entity_emb = entity_embedding_layer(entity_input)
    # entity_emb = Dropout(0.2)(entity_emb)

    title_co_emb = Self_Attention(20, 20)([title_emb, entity_emb, entity_emb])
    entity_co_emb = Self_Attention(20, 20)([entity_emb, title_emb, title_emb])

    title_vecs = Self_Attention(20, 20)([title_emb, title_emb, title_emb])
    title_vector = keras.layers.Add()([title_vecs, title_co_emb])

    title_vector = Dropout(0.2)(title_vector)
    title_vec = AttLayer(400, seed)(title_vector)

    entity_vecs = Self_Attention(20, 20)([entity_emb, entity_emb, entity_emb])
    entity_vector = keras.layers.Add()([entity_vecs, entity_co_emb])

    entity_vector = Dropout(0.2)(entity_vector)
    entity_vec = AttLayer(400, seed)(entity_vector)

    feature = [title_vec, entity_vec]
    news_vec = keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(feature)
    news_vec = Dropout(0.2)(news_vec)
    news_vecs = AttLayer(400, seed)(news_vec)

    model = Model(news_input, news_vecs)
    return model


def get_popularity_encoder(config, seed, t):
    LengthTable = {'title': config['title_length'],
                   'entity': config['max_entity_num']}
    input_length = 0
    PositionTable = {}
    for v in config['attrs']:
        PositionTable[v] = (input_length, input_length + LengthTable[v])
        input_length += LengthTable[v]
    print(PositionTable)

    entity_popularity_embedding_layer = Embedding(200, 200, trainable=True)
    time_embedding_layer = Embedding(505, 100, trainable=True)

    all_input = Input((input_length + 1,), dtype='float32')
    news_input = keras.layers.Lambda(lambda x: x[:, :input_length])(all_input)
    news_time_input = keras.layers.Lambda(lambda x: x[:, input_length:input_length + 1])(all_input)

    title_input = keras.layers.Lambda(lambda x: x[:, PositionTable['title'][0]:PositionTable['title'][1]])(news_input)
    entity_input = keras.layers.Lambda(lambda x: x[:, PositionTable['entity'][0]:PositionTable['entity'][1]])(
        news_input)

    title_emb = entity_popularity_embedding_layer(title_input)
    title_emb = Dropout(0.2)(title_emb)

    entity_emb = entity_popularity_embedding_layer(entity_input)
    entity_emb = Dropout(0.2)(entity_emb)

    title_co_emb = Self_Attention(400, 1)([title_emb, entity_emb, entity_emb])
    entity_co_emb = Self_Attention(400, 1)([entity_emb, title_emb, title_emb])

    title_vecs = Self_Attention(400, 1)([title_emb, title_emb, title_emb])
    title_vector = keras.layers.Add()([title_vecs, title_co_emb])

    title_vector = Dropout(0.2)(title_vector)
    title_vec = AttLayer(400, seed)(title_vector)

    entity_vecs = Self_Attention(400, 1)([entity_emb, entity_emb, entity_emb])
    entity_vector = keras.layers.Add()([entity_vecs, entity_co_emb])

    entity_vector = Dropout(0.2)(entity_vector)
    entity_vec = AttLayer(400, seed)(entity_vector)

    feature = [title_vec, entity_vec]
    news_vec = keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(feature)
    news_vec = Dropout(0.2)(news_vec)
    news_vecs = AttLayer(400, seed)(news_vec)

    vec1 = Dense(256, activation='tanh')(news_vecs)
    vec1 = Dense(256)(vec1)
    vec1 = Dense(128, )(vec1)
    popularity_score = Dense(1, activation='sigmoid')(vec1)
    time_emb = time_embedding_layer(news_time_input)
    vec2 = Dense(64, activation='tanh')(time_emb)
    vec2 = Dense(64)(vec2)
    popularity_recency_score = Dense(1, activation='sigmoid')(vec2)
    popularity_recency_score = tf.keras.layers.Reshape((1,))(popularity_recency_score)

    popularity_fusion_score = keras.layers.Lambda(lambda x: x[0] * ((1/x[1])**t))([popularity_score, popularity_recency_score])

    model = Model(all_input, popularity_fusion_score)
    return model


def popularity_fusion(config):
    LengthTable = {'title': config['title_length'],
                   'entity': config['max_entity_num']}
    input_length = 0
    PositionTable = {}
    for v in config['attrs']:
        PositionTable[v] = (input_length, input_length + LengthTable[v])
        input_length += LengthTable[v]
    print(PositionTable)
    news_input_1 = Input((input_length,), dtype='int32')
    news_input_2 = Input((1,), dtype='int32')
    popularity_vec = keras.layers.Concatenate(axis=-1)([news_input_1, news_input_2])
    model = Model([news_input_1, news_input_2], popularity_vec)
    return model


def create_pe_model(config, model_config, News, word_embedding_matrix, entity_embedding_matrix, seed):
    max_clicked_news = config['max_clicked_news']
    t = model_config['popularity_time']
    entity_embedding_layer = Embedding(entity_embedding_matrix.shape[0], entity_embedding_matrix.shape[1],
                                       weights=[entity_embedding_matrix], trainable=True)
    user_embedding_layer = Embedding(len(News.dg.user_index) + 1, 400, trainable=True)

    popularity_encoder = get_popularity_encoder(config, seed, t)
    news_encoder = get_news_encoder_co1(config, len(News.word_dict), word_embedding_matrix, entity_embedding_layer, seed)
    popularity_fusion_encoder = popularity_fusion(config)

    news_input_length = int(news_encoder.input.shape[1])

    clicked_input = Input(shape=(max_clicked_news, news_input_length,), dtype='int32')    # input_3
    uid = Input(shape=(1, ), dtype='int32')
    candidates = keras.Input((1 + config['np_ratio'], news_input_length,), dtype='int32')    # input_5
    clicked_buckets = keras.Input((1 + config['np_ratio'], news_input_length,), dtype='float32')  # input_6
    news_time = keras.Input((1 + config['np_ratio'], ), dtype='float32')
    click_ctr = Input(shape=(max_clicked_news,), dtype='int32')

    user_vecs = TimeDistributed(news_encoder)(clicked_input)     # connected to input_3
    uid_vecs = tf.keras.layers.Reshape((400,))(user_embedding_layer(uid))

    if config['user_encoder_name'] == 'SelfAtt':
        user_vecs = Self_Attention(20, 20)([user_vecs, user_vecs, user_vecs])    # connected to time_dis0
        user_vecs = Dropout(0.2)(user_vecs)
        user_vec = AttentivePooling(max_clicked_news, 400)(user_vecs)
    elif config['user_encoder_name'] == 'Att':
        user_vecs = Dropout(0.2)(user_vecs)
        user_vec = AttentivePooling(max_clicked_news, 400)(user_vecs)
    elif config['user_encoder_name'] == 'GRU':
        user_vec = GRU(400, activation='tanh')(
            tf.keras.layers.Masking(mask_value=0.0)(user_vecs),
            initial_state=[uid_vecs], )
    elif config['user_encoder_name'] == 'GRU-attention':
        user_vecs = GRU(400, activation='tanh', return_sequences=True)(
            tf.keras.layers.Masking(mask_value=0.0)(user_vecs),
            initial_state=[uid_vecs], )
        user_vecs = Self_Attention(20, 20)([user_vecs, user_vecs, user_vecs])
        user_vec = Dropout(0.2)(user_vecs)
        user_vec = AttentivePooling(max_clicked_news, 400)(user_vec)
    elif config['user_encoder_name'] == 'SelfAtt-GRU':
        user_vecs = Self_Attention(20, 20)([user_vecs, user_vecs, user_vecs])
        user_vec = GRU(400, activation='tanh')(
            tf.keras.layers.Masking(mask_value=0.0)(user_vecs),
            initial_state=[uid_vecs], )
    elif config['user_encoder_name'] == 'SelfAtt-GRU-ini':
        user_vector = Self_Attention(20, 20)([user_vecs, user_vecs, user_vecs])
        user_vector = Dropout(0.2)(user_vector)
        user_vector = AttentivePooling(max_clicked_news, 400)(user_vector)
        user_vec = GRU(400, activation='tanh')(
            tf.keras.layers.Masking(mask_value=0.0)(user_vecs),
            initial_state=[user_vector], )
    elif config['user_encoder_name'] == 'popularity_user_modeling':
        popularity_embedding_layer = Embedding(200, 400, trainable=True)
        popularity_embedding = popularity_embedding_layer(click_ctr)
        MHSA = Self_Attention(20, 20)
        user_vecs = MHSA([user_vecs, user_vecs, user_vecs])
        user_vec_query = keras.layers.Concatenate(axis=-1)([user_vecs, popularity_embedding])
        user_vec = AttentivePoolingQKY(50, 800, 400)([user_vec_query, user_vecs])

    candidate_vecs = TimeDistributed(news_encoder)(candidates)
    rel_scores = keras.layers.Dot(axes=-1)([user_vec, candidate_vecs])

    news_times = tf.keras.layers.Reshape((1 + config['np_ratio'], 1,))(news_time)
    popularity_vec = keras.layers.Concatenate(axis=-1)([clicked_buckets, news_times])
    popularity_score = TimeDistributed(popularity_encoder)(popularity_vec)
    popularity_scores = tf.keras.layers.Reshape((1 + config['np_ratio'],))(popularity_score)

    user_vec_input = keras.layers.Input((400,), )
    activity_gate = Dense(128, activation='tanh')(user_vec_input)
    activity_gate = Dense(64)(activity_gate)
    activity_gate = Dense(1, activation='sigmoid')(activity_gate)
    activity_gate = keras.layers.Reshape((1,))(activity_gate)
    activity_gater = Model(user_vec_input, activity_gate)
    user_activtiy = activity_gater(user_vec)

    scores = []
    rel_scores = keras.layers.Lambda(lambda x: 2 * x[0] * x[1])([rel_scores, user_activtiy])
    scores.append(rel_scores)
    bias_score = keras.layers.Lambda(lambda x: 2 * x[0] * (1 - x[1]))([popularity_scores, user_activtiy])
    scores.append(bias_score)
    scores = keras.layers.Add()(scores)

    logits = keras.layers.Activation(keras.activations.softmax, name='recommend')(scores)
    model = Model([candidates, clicked_input, clicked_buckets, news_time, uid, click_ctr], [logits])

    # categorical_crossentropy
    # BinaryCrossentropy
    # PairwiseLogisticLoss
    # metrics = [tf.keras.metrics.AUC(), tfr.keras.metrics.MRRMetric(), tfr.keras.metrics.NDCGMetric(topn=5),
    #            tfr.keras.metrics.NDCGMetric(topn=10)]
    # tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  metrics=[tf.keras.metrics.Accuracy()])

    user_encoder = Model([clicked_input, click_ctr], user_vec)
    return model, user_encoder, news_encoder, activity_gater, popularity_encoder, popularity_fusion_encoder
