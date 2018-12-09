from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, LSTM, Embedding, Lambda, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.models import Model

def siamese_LSTM_w_sigmoid():
	embedding_layer = Embedding(len(embedding_matrix), 300, weights=[embedding_matrix], 
		input_length=max_sentence_length, trainable=False)
	lstm_layer = LSTM(128)

	question_1_input = Input(shape=(max_sentence_length, ), dtype='int32')
	question_1_embedded = embedding_layer(question_1_input)
	question_1_vec = lstm_layer(question_1_embedded)

	question_2_input = Input(shape=(max_sentence_length, ), dtype='int32')
	question_2_embedded = embedding_layer(question_2_input)
	question_2_vec = lstm_layer(question_2_embedded)

	distance = Lambda(lambda x: K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True))([question_1_vec, question_2_vec])
	dense_1 = Dense(16, activation='sigmoid')(distance)
	dense_1 = Dropout(0.3)(dense_1)
	batch_normal_1 = BatchNormalization()(dense_1)
	prediction = Dense(1, activation='sigmoid')(batch_normal_1)

	model = Model(input=[question_1_input, question_2_input], output=prediction)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

	early_stopping =EarlyStopping(monitor='val_loss', patience=3)
	model.fit([question1_train, question2_train], y_train, validation_data=([question1_val, question2_val], y_val), verbose=1, 
          nb_epoch=10, batch_size=256, shuffle=True,class_weight=None, callbacks=[early_stopping])

	pred = model.predict([question1_test, question2_test], verbose=1)