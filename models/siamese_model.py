from keras.layers import Dense, Input, LSTM, Embedding, Lambda, Dropout, BatchNormalization, Bidirectional
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras import backend as K
from keras.models import Model
import numpy as np

max_sentence_length = 25

def train(logger, X_train, X_val, X_test, y_train, y_val, y_test, embedding_matrix, distance_type, bidirectional):
	question1_train = X_train[:,0]
	question2_train = X_train[:,1]
	question1_val = X_val[:,0]
	question2_val = X_val[:,1]
	question1_test = X_test[:,0]
	question2_test = X_test[:,1]

	csv_logger = CSVLogger('logs/log.csv', append=True, separator=';')
	model_name = "siamese_" + distance_type + ".h5"
	prediction_name = "pred_siamese_" + distance_type + ".npy"

	embedding_layer = Embedding(len(embedding_matrix), 300, weights=[embedding_matrix], 
		input_length=max_sentence_length, trainable=False)
	if bidirectional:
		lstm_layer = Bidirectional(128)
	else: 
		lstm_layer = LSTM(128)

	question_1_input = Input(shape=(max_sentence_length, ), dtype='int32')
	question_1_embedded = embedding_layer(question_1_input)
	question_1_vec = lstm_layer(question_1_embedded)

	question_2_input = Input(shape=(max_sentence_length, ), dtype='int32')
	question_2_embedded = embedding_layer(question_2_input)
	question_2_vec = lstm_layer(question_2_embedded)

	if distance_type == "manhattan":
		distance = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)))([question_1_vec, question_2_vec])
	elif distance_type == "euclidean":
		distance = Lambda(lambda x: K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True))([question_1_vec, question_2_vec])
	elif distance_type == "cosine": 
		distance = Lambda(lambda x: -K.mean((K.l2_normalize(x[0], axis=-1) * K.l2_normalize(x[1], axis=-1)), axis=-1, keepdims=True))([question_1_vec, question_2_vec])
	dense_1 = Dense(16, activation='sigmoid')(distance)
	dense_1 = Dropout(0.3)(dense_1)
	batch_normal_1 = BatchNormalization()(dense_1)
	prediction = Dense(1, activation='sigmoid')(batch_normal_1)

	model = Model(input=[question_1_input, question_2_input], output=prediction)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
	model.summary(print_fn=lambda x: logger.info(x))

	early_stopping =EarlyStopping(monitor='val_loss', patience=3)
	model.fit([question1_train, question2_train], y_train, validation_data=([question1_val, question2_val], y_val), verbose=1, 
          nb_epoch=10, batch_size=256, shuffle=True,class_weight=None, callbacks=[early_stopping, csv_logger])

	model.save(model_name)

	pred = model.predict([question1_test, question2_test], verbose=1)
	np.save(open(prediction_name, 'wb'), pred)
	logger.info(f"Correct predction count: {sum(y_test == pred)}")
	logger.info(f"Test length: {len(y_test)}")