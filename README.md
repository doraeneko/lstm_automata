# Learning regular languages using recurrent neural networks

## Short Description

This repo contains a demonstrator for learning acceptors of (mostly regular) languages
using an RNN acceptor that, given a word, decides whether it belongs to the language or not (binary
classifier). The approach uses LSTM neurons (with peepholes) and requires a "language oracle" that
decides whether an arbitrary word belongs to the language or not. It uses positive and negative examples for 
the training. 
As an example, the first five Tomita languages are added; training works quite well for these.

## Used frameworks

TensorFlow and Keras.

## How to use it
Take a look at 

	$ test_tomita.py

- define an arbitrary language alphabet where letters correspond to chars (e.g. `['a', 'b']`), e.g. `alphabet = Alphabet(["0", "1"])`
- define a language; it takes a name, an alphabet instance and a language oracle (Python callable). The oracle takes a word
  and returns `True` iff the word is element of the language you want to learn (third parameter of the constructor
of `Language`, e.g. ` lambda word: len(word) >= 1 and word.find("0") == -1`)
- define a net topology by specifying an instance of `NeuralNetTopology`; a net is always structured as follows:
	 1. input layer; its size corresponds to the number of alphabet letters (one-hot-encoding of letters)
	 2. 0 or more dense network layers: specify number and size of layers with `dense_layers_before_lstm`. E.g., `dense_layers_before_lstm=[50, 10]` will build two layers, first 50 units, second 10 units
	 3. 1 or more LSTM layers: specify using `lstm_layers`
	 4. 0 or more dense layers: specify using `dense_layers_after_lstm`
	 5. final layer with two outputs (accepting and not accepting outcomes)
- create an `LSTMAutomataLearner` instance: `learner = LSTMAutomataLearner(language, topology)`
- train the model using `learner.train()`.

