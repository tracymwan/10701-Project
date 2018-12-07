#import keras, tensorflow as tf
import numpy as np, pandas as pd
import logging, os.path, sys, time, random
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger("10701")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')

logs_dir = "logs"
data_dir = "questions.csv"

def parse_arguments(): 
	parser = argparse.ArgumentParser(description='Identifying Duplicate Question Parser')

def preprocess():
	data = pd.read_csv('questions.csv')

	data['question1'] = data['question1'].apply(lambda x: (str(x)))
	data['question2'] = data['question2'].apply(lambda x: (str(x)))
	question1 = list(data['question1'])
	question2 = list(data['question2'])

	target = np.array(data['is_duplicate'], dtype=int)

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(question1+question2)

	question1_word_sequences = tokenizer.texts_to_sequences(question1)
	q1_data = pad_sequences(question1_word_sequences, maxlen=25)
	question2_word_sequences = tokenizer.texts_to_sequences(question2)
	q2_data = pad_sequences(question2_word_sequences, maxlen=25)

	X = np.stack((q1_data, q2_data), axis = 1)
	X, X_test, y, y_test = train_test_split(X, target, test_size = 0.25, stratify=target)
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, stratify=y)

	q1_train = X_train[:,0]
	q2_train = X_train[:,1]
	q1_val = X_val[:,0]
	q2_val = X_val[:,1]
	q1_train = X_test[:,0]
	q2_train = X_test[:,1]


def main(args):
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

	preprocess()

	logger.info(f"Log saved to {log_file_name}")


if __name__ == '__main__':
	preprocess()