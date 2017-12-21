Requirements (version we used):
Keras (version 2.0.8)
Tensorflow (versioin 1.3.0)

Usage:

Training:
	python train.py
	
This will use all files in "data/*.txt" for training. 
Adjust number of training epochs inside the file. 
Will output a file "trained_<timestamp>.h5" when finished, as well as a file every epoch.
	
Prediction:
	python -i predict.py <path to trained file>
	
This will open python in interactive mode and load the provided model. 
To generate texts, use the function: 

	gen(seed, n, top_n = 5, include_seed = False, return_probs = False)
	
seed = the string text from which the generator will generate.
n = number of characters to generate
top_n = number of characters to choose from at each step. top_n=1 will always choose the most probable character. top_n=x will choose from the top x most probable characters
include_seed = bool, if true, the returned string will include the provided seed at the start.
return_probs = should the function return the probability vector for each step? If true, the output of the function is a tuple: (string,list[dict]). otherwise the output is just a string. If include_seed is true, note that the indexes of the probabilitis list will not match the string.