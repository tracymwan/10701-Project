from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, \
            Bidirectional, TimeDistributed, Dropout, Conv1D, GlobalMaxPool1D, merge, BatchNormalization
from keras.layers.merge import multiply, concatenate
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras import backend as K
from keras.models import Model
import numpy as np

max_sentence_length = 25

def train(logger, X_train, X_val, X_test, y_train, y_val, y_test, embedding_matrix, distance_type):
    question1_train = X_train[:, 0]
    question2_train = X_train[:, 1]
    question1_val = X_val[:, 0]
    question2_val = X_val[:, 1]
    question1_test = X_test[:, 0]
    question2_test = X_test[:, 1]

    csv_logger = CSVLogger('logs/log.csv', append=True, separator=';')

    embedding_layer = Embedding(
        len(embedding_matrix),
        300,
        weights=[embedding_matrix],
        input_length=max_sentence_length,
        trainable=False)

    # lstm_layer = LSTM(128)
    n_hidden = 128

    question_1_input = Input(shape=(max_sentence_length, ), dtype='int32')
    question_1_embedded = embedding_layer(question_1_input)
    activations = Bidirectional(
        LSTM(n_hidden, return_sequences=True),
        merge_mode='concat')(question_1_embedded)
    question_1_vec = Bidirectional(
        LSTM(n_hidden, return_sequences=True),
        merge_mode='concat')(activations)

    # question_1_vec = lstm_layer(question_1_embedded)

    attention1 = TimeDistributed(Dense(1, activation='tanh'))(question_1_vec)
    attention1 = Flatten()(attention1)
    attention1 = Activation('softmax')(attention1)
    attention1 = RepeatVector(n_hidden * 2)(attention1)
    attention1 = Permute([2, 1])(attention1)
    sent_representation1 = multiply([question_1_vec, attention1])
    sent_representation1 = Lambda(lambda xin: K.sum(xin, axis=1))(
        sent_representation1)

    question_2_input = Input(shape=(max_sentence_length, ), dtype='int32')
    question_2_embedded = embedding_layer(question_2_input)
    # question_2_vec = lstm_layer(question_2_embedded)

    activations_2 = Bidirectional(
        LSTM(n_hidden, return_sequences=True),
        merge_mode='concat')(question_2_embedded)
    question_2_vec = Bidirectional(
        LSTM(n_hidden, return_sequences=True),
        merge_mode='concat')(activations_2)

    attention2 = TimeDistributed(Dense(1, activation='tanh'))(question_2_vec)
    attention2 = Flatten()(attention2)
    attention2 = Activation('softmax')(attention2)
    attention2 = RepeatVector(n_hidden * 2)(attention2)
    attention2 = Permute([2, 1])(attention2)
    sent_representation2 = multiply([question_2_vec, attention2])
    sent_representation2 = Lambda(lambda xin: K.sum(xin, axis=1))(
        sent_representation2)

    if distance_type == "manhattan":
        distance = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)))(
            [sent_representation1, sent_representation2])
    elif distance_type == "euclidean":
        distance = Lambda(
        lambda x: K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True))(
            [sent_representation1, sent_representation2])
    elif distance_type == "cosine": 
        distance = Lambda(lambda x: -K.mean((K.l2_normalize(x[0], axis=-1) * K.l2_normalize(x[1], axis=-1)), axis=-1, keepdims=True))(
            [sent_representation1, sent_representation2])
    dense_1 = Dense(16, activation='sigmoid')(distance)
    dense_1 = Dropout(0.3)(dense_1)
    batch_normal_1 = BatchNormalization()(dense_1)
    prediction = Dense(1, activation='sigmoid')(batch_normal_1)

    model = Model(
        input=[question_1_input, question_2_input], output=prediction)
    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary(print_fn=lambda x: logger.info(x))

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(
        [question1_train, question2_train],
        y_train,
        validation_data=([question1_val, question2_val], y_val),
        verbose=1,
        epochs=10,
        batch_size=256,
        shuffle=True,
        class_weight=None,
        callbacks=[early_stopping, csv_logger])

    pred = model.predict([question1_test, question2_test], verbose=1)
    np.save(open('pred.npy', 'wb'), pred)
    logger.info(f"Correct predction count: {sum(y_test == pred)}")
    logger.info(f"Test length: {len(y_test)}")