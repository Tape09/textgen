# import numpy
import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import LSTM
# from keras.callbacks import ModelCheckpoint
# from keras.utils import np_utils
import numpy as np
import time;
from tqdm import tqdm
import sys
import math
import glob
import os

# chars = [' ', '!', '"',"'", '&', '(', ')', '*', ',', '-', '.', ':', ';', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
chars = [' ', '!', '"',"'", ',', '-', '.', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# for i in range(10):
	# chars.append(str(i));
	
char_to_vec = dict();

for i,c in enumerate(chars):
	char_to_vec[c] = keras.utils.to_categorical(i,len(chars));

def read_book(filename):
	raw_text = open(filename).read()
	raw_text = raw_text.lower()
	sz = 0;
	vec_text = [None] * len(raw_text);
	for c in raw_text:
		if c in char_to_vec:
			vec_text[sz] = np.reshape(char_to_vec[c],(1,1,len(chars)));
			sz += 1;
	return vec_text[:sz];
	
def vec_to_char(vec,top_n = 1):
	# print(vec.shape)
	top_idxs = np.argsort(vec[0,:])[-top_n:];
	top_probs = vec[0,0,top_idxs];
	# np.reshape(top_probs,(len(chars),))
	# print(top_idxs.shape)
	# print(top_probs.shape)
	
	top_probs = top_probs / np.sum(top_probs[0]);
	idx = np.random.choice(top_idxs[0],p=top_probs[0])

	return chars[idx];

# print("Reading books");
	
# load ascii text and covert to lowercase
books = [];

filenames = glob.glob("data/*.txt")
for filename in filenames:
	books.append(read_book(filename));


for book in books:
	n = len(book);
	idx = np.random.randint(n-1000);
	s = "";
	for i in range(150):
		s+=vec_to_char(book[idx + i]);
	print(s);
	print()
	
	
	
	
	