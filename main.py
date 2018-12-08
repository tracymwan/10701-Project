import keras, tensorflow as tf
import numpy as np, pandas as pd
import logging, os.path, sys, time, random, argparse
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Lambda

logger = logging.getLogger("10701")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')

logs_dir = "logs"
data_dir = "questions.csv"
glove_dir = "glove.840B.300d.txt"

max_sentence_length = 25

def parse_arguments(): 
	parser = argparse.ArgumentParser(description='Identifying Duplicate Question Parser')
	parser.add_argument('--preprocess', dest='preprocess', action="store_true")
	return parser.parse_args()

def preprocess():
	# preparing the text data
	data = pd.read_csv('questions.csv')

	data['question1'] = data['question1'].apply(lambda x: (str(x)))
	data['question2'] = data['question2'].apply(lambda x: (str(x)))
	question1 = list(data['question1'])
	question2 = list(data['question2'])

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(question1+question2)
	question1_word_sequences = tokenizer.texts_to_sequences(question1)
	question1_data = pad_sequences(question1_word_sequences, maxlen=max_sentence_length)
	question2_word_sequences = tokenizer.texts_to_sequences(question2)
	question2_data = pad_sequences(question2_word_sequences, maxlen=max_sentence_length)
	word_index = tokenizer.word_index
	logger.info(f"Found %s unique tokens." % len(word_index))

	target = np.array(data['is_duplicate'], dtype=int)

	logger.info(f"Shape of data tensor:", question1_data.shape)
	logger.info(f"Shape of label tensor:", target)

	# preparing the embedding layer
	embeddings_index = {}
	f = open(glove_dir)
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	logger.info(f"Found %s word vectors." % len(embeddings_index))

	EMBEDDING_DIM = 300
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embeddings_vector = embeddings_index.get(word)
		if embeddings_vector is not None: 
			embedding_matrix[i] = embeddings_vector

	np.save(open('q1_train.npy', 'wb'), q1_train)
	np.save(open('q2_train.npy', 'wb'), q2_train)
	np.save(open('target.npy', 'wb'), target)
	np.save(open('embedding_matrix.npy', 'wb'), embedding_matrix)

def train():
	question1_data = np.load(open('q1_train.npy', 'rb'))
	question2_data = np.load(open('q2_train.npy', 'rb'))
	target = np.load(open('target.npy', 'rb'))
	embedding_matrix = np.load(open('embedding_matrix.npy', 'rb'))

	X = np.stack((question1_data, question2_data), axis = 1)
	X, X_test, y, y_test = train_test_split(X, target, test_size = 0.15, stratify=target)
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.15, stratify=y)

	question1_train = X_train[:,0]
	question2_train = X_train[:,1]
	question1_val = X_val[:,0]
	question2_val = X_val[:,1]
	question1_test = X_test[:,0]
	question2_test = X_test[:,1]

	embedding_layer = Embedding(len(embedding_matrix), 300, weights=[embedding_matrix], 
		input_length=max_sentence_length, trainable=False)
	lstm_layer = LSTM(128)

	question_1_input = Input(shape=(max_sentence_length, ), dtype='int32')
	question_1_embedded = embedding_layer(question_1_input)
	question_1_vec = lstm_layer(question_1_embedded)

	question_2_input = Input(shape=(max_sentence_length, ), dtype='int32')
	question_2_embedded = embedding_layer(question_2_input)
	question_2_vec = lstm_layer(question_2_embedded)

	distance = Lambda(lambda x, y: K.sum(K.square(x - y), axis=1, keepdims=True))([question_1_vec, question_2_vec])
	dense_1 = Dense(16, activation='sigmoid')(distance)
	dense_1 = Dropout(0.3)(dense1)
	batch_normal_1 = BatchNormalization()(dense_1)
	prediction = Dense(1, activation='sigmoid')(batch_normal_1)

	model = Model(input=[question_1_input, question_2_input], output=prediction)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

	early_stopping =EarlyStopping(monitor='val_loss', patience=3)
	model.fit([Q1_train, Q2_train], y_train, validation_data=([Q1_val, Q2_val], y_val), verbose=1, 
          nb_epoch=10, batch_size=256, shuffle=True,class_weight=None, callbacks=[early_stopping])

	pred = model.predict([question1_test, question2_test], verbose=1)


def main(args): 
	args = parse_arguments()

	if not os.path.exists(logs_dir):
		os.makedirs(logs_dir)

	# logging settings
	os.environ['TZ'] = 'EST+05EDT,M4.1.0,M10.5.0'
	time.tzset()

	time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
	log_file_name = f'{logs_dir}/{time_str}.log'
	hdlr = logging.FileHandler(log_file_name)
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr)
	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(logging.DEBUG)
	ch.setFormatter(formatter)
	logger.addHandler(ch)

    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

    # Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	logger.info(f"Command line args: {args}")
	logger.info(f"Log saving to {log_file_name}")

	time_seed = int(''.join(time_str.split('_'))) % (2 ** 32)
	np.random.seed(time_seed)
	logger.info(f"Numpy random seed {time_seed}")

	if args.preprocess:
		preprocess()

	logger.info(f"Log saved to {log_file_name}")


if __name__ == '__main__':
	main(sys.argv)