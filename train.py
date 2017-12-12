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

def create_model(batch_size = 1):
	model = Sequential()
	model.add(LSTM(256, batch_input_shape=(batch_size,1,len(chars)),stateful=True, return_sequences = True))
	model.add(Dropout(0.2))
	model.add(LSTM(256,stateful=True,return_sequences = True))
	model.add(Dropout(0.2))
	model.add(LSTM(256,stateful=True,return_sequences = True))
	model.add(Dropout(0.2))
	model.add(Dense(len(chars), activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model;

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
	
def vec_to_char(vec):
	return np.argmax(vec);
	

print("Reading books");
	
# load ascii text and covert to lowercase
books = [];

filenames = glob.glob("data/*.txt")
for filename in filenames:
	books.append(read_book(filename));


savefile_name = time.strftime("%Y%m%d-%H%M");

print("Building model")
n_batches = 256;
letters_per_seq = 1000;
model = create_model(n_batches);
	
# sys.exit()
	
n_epochs = 120;

loss = [];
for epoch in range(n_epochs):
	print("Epoch:",epoch+1,"/",n_epochs)
	# mean_tr_acc = []
	mean_tr_loss = []
	for b in range(len(books)):
		print("\tbook:",b+1,"/",len(books))
		
		book_len = len(books[b])
		krange = int(math.ceil(len(books[b]) / (n_batches * letters_per_seq)))
		for k in tqdm(range(krange)):
			# print("\t\tstartpoint:",k+1,"/",krange)	
			starts = [np.random.randint(book_len-letters_per_seq) for i in range(n_batches)]
			x = books[b][starts[0]:starts[0]+letters_per_seq];
			for i in range(1,n_batches):
				xtemp = books[b][starts[i]:starts[i]+letters_per_seq];
				for j in range(letters_per_seq):
					x[j] = np.concatenate((x[j],xtemp[j]))
			
			for i in range(letters_per_seq-1):
				tr_loss = model.train_on_batch(x[i], x[i+1]);
				mean_tr_loss.append(tr_loss)
				
			model.reset_states()
			# model.save(savefile_name+"_temp.hdf5")
	
	loss.append(np.mean(mean_tr_loss));
	model.save(savefile_name+"_"+str(epoch)+".hdf5")
		

model.save("trained_"+savefile_name+".h5")
np.save("loss_"+savefile_name+".npy",loss)
