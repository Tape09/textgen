import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np
import time;
from tqdm import tqdm
import sys


# chars = [' ', '!', '"',"'", '&', '(', ')', '*', ',', '-', '.', ':', ';', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# for i in range(10):
	# chars.append(str(i));
	
chars = [' ', '!', '"',"'", ',', '-', '.', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

	
char_to_vec = dict();

def char_to_vecc(c):
	return np.reshape(char_to_vec[c],(1,1,len(chars)));

for i,c in enumerate(chars):
	char_to_vec[c] = keras.utils.to_categorical(i,len(chars));

def max_char_index(vec):
	return np.argmax(vec);

def vec_to_char(vec,top_n = 1):
	# print(vec.shape)
	top_idxs = np.argsort(vec[0,:])[-top_n:];
	top_probs = vec[0,top_idxs];
	
	top_probs = top_probs / np.sum(top_probs);
	idx = np.random.choice(top_idxs,p=top_probs)

	return chars[idx];
	
def load_model(filename):
	return keras.models.load_model(filename);

def generate(model,n,seed=". ", top_n = 1, include_seed = True):
	batch_size = model.inputs[0].get_shape().as_list()[0];
	model.reset_states();
	for c in seed:
		y = model.predict(np.repeat(char_to_vecc(c),batch_size,axis=0),batch_size=batch_size);
		
	next_char = vec_to_char(y[0],top_n);
	if(include_seed):
		word = seed + next_char;
	else:
		word = "" + next_char;
		
	for i in range(n):
		# c = vec_to_char(y[0],top_n);
		y = np.repeat(char_to_vecc(next_char),batch_size,axis=0)
		y = model.predict(y,batch_size=batch_size);
		next_char = vec_to_char(y[0],top_n);
		word += next_char;
		
	return word;
	
	
def prob_dict(y):
	d = dict();
	for i in range(y.shape[1]):
		d[chars[i]] = round(y[0,i],3);
	return d;
	
filename = sys.argv[1];
model = load_model(filename);


def gen(seed, n, top_n = 5, include_seed = False, return_probs = False):
	batch_size = model.inputs[0].get_shape().as_list()[0];
	model.reset_states();
	seed = seed.lower();
	for c in seed:
		y = model.predict(np.repeat(char_to_vecc(c),batch_size,axis=0),batch_size=batch_size);
		
	probs = [];
	probs.append(prob_dict(y[0]));
	next_char = vec_to_char(y[0],top_n);
	if(include_seed):
		word = seed + next_char;
	else:
		word = "" + next_char;
		
	for i in range(n):
		# c = vec_to_char(y[0],top_n);
		y = np.repeat(char_to_vecc(next_char),batch_size,axis=0)
		y = model.predict(y,batch_size=batch_size);
		probs.append(prob_dict(y[0]));
		next_char = vec_to_char(y[0],top_n);
		word += next_char;
		
	if return_probs:
		return word,probs;
	else:
		return word
	
	
	









