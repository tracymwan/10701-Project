import keras, tensorflow as tf, numpy as np
import logging, os.path, sys, time, random, argparse
from sklearn.model_selection import train_test_split
import models.siamese_model as siamese_model
import models.attention_model as attention_model
import preprocess


logger = logging.getLogger("10701")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')

logs_dir = "logs"

def parse_arguments(): 
	parser = argparse.ArgumentParser(description='Identifying Duplicate Question Parser')
	parser.add_argument('--preprocess', dest='preprocess', action="store_true")
	parser.add_argument('--model_type', dest='model_type', type=str, help="siamese or attention")
	parser.add_argument('--distance_type', dest='distance_type', type=str, help="manhattan or euclidean")
	parser.add_argument('--random_seed', dest='random_seed', type=int)
	return parser.parse_args()

def run_model(model_type, distance_type):
	question1_data = np.load(open('q1_train.npy', 'rb'))
	question2_data = np.load(open('q2_train.npy', 'rb'))
	target = np.load(open('target.npy', 'rb'))
	embedding_matrix = np.load(open('embedding_matrix.npy', 'rb'))

	X = np.stack((question1_data, question2_data), axis = 1)
	X, X_test, y, y_test = train_test_split(X, target, test_size = 0.15, stratify=target)
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.15, stratify=y)

	if model_type == "siamese":
		siamese_model.train(logger, X_train, X_val, X_test, y_train, y_val, y_test, embedding_matrix, distance_type)
	elif model_type == "attention":
		attention_model.train(logger, X_train, X_val, X_test, y_train, y_val, y_test, embedding_matrix, distance_type)

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
	if (args.random_seed):
		np.random.seed(args.random_seed)
	else:
		np.random.seed(time_seed)
	logger.info(f"Numpy random seed {time_seed}")

	if args.preprocess:
		preprocess.preprocess_data()
	run_model(args.model_type, args.distance_type)

	logger.info(f"Log saved to {log_file_name}")


if __name__ == '__main__':
	main(sys.argv)