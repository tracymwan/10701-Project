import numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

data_dir = "questions.csv"
glove_dir = "glove.840B.300d.txt"
max_sentence_length = 25

def preprocess_data():
	# preparing the text data
	data = pd.read_csv(data_dir)

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

	target = np.array(data['is_duplicate'], dtype=int)

	# preparing the embedding layer
	embeddings_index = {}
	f = open(glove_dir)
	for line in f:
		values = line.split()
		if (len(values) != 301):
			continue
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()

	EMBEDDING_DIM = 300
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embeddings_vector = embeddings_index.get(word)
		if embeddings_vector is not None: 
			embedding_matrix[i] = embeddings_vector

	np.save(open('q1_train.npy', 'wb'), question1_data)
	np.save(open('q2_train.npy', 'wb'), question2_data)
	np.save(open('target.npy', 'wb'), target)
	np.save(open('embedding_matrix.npy', 'wb'), embedding_matrix)