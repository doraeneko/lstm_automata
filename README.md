# Learning regular languages using recurrent neural networks


## Description
-----

Python script to train approximative acceptors of regular languages based on recurrent neural networks.
The resulting acceptor takes a word as input and decides whether it belongs to the language or not. 
The approach uses LSTM neurons with peepholes and requires for the training a "language oracle" that
decides whether an arbitrary word belongs to the language or not. It uses positive and negative examples for 
the training (quite a lot of them, most of the cases ca. 1000 words). 


As an example, the seven Tomita languages are added; these are often used for benchmarking automata learning techniques.
Training works quite well for these and requires
rather few neurons. The deduced networks in `test_tomita.py` 
appear to be even often equivalent to the corresponding finite automata of the Tomita languages (loss 0 and correct 
recognition of all words up to a given maximal size, at least in my setting). Checking whether the acceptors in fact 
exactly recognize the corresponding Tomita language is future work.

## Used neural net topology
-----
Dense Layers are followed by LSTM layers, then dense layers again, with two output neurons.
The fine-grained topology of the neural net can be customized by specifying number of layers and number of neurons in these layers. 
The peephole connection extension [Gers, Schmidhuber: Recurrent nets that time and count](https://www.researchgate.net/publication/3857862_Recurrent_nets_that_time_and_count) to the LSTM neuron proved to be crucial
for the training to be successful - nets using standard LSTM neurons from tensorflow seem not to converge satisfactorily in this setting;
the used LSTM cell is [`tfa.rnn.PeepholeLSTMCell`](https://www.tensorflow.org/addons/api_docs/python/tfa/rnn/PeepholeLSTMCell).

## Used frameworks
-----

TensorFlow and Keras. You will need the Python packages `numpy`, `tensorflow`, `tensorflow_addons` and `keras`.

## How to use it
-----

Take a look at 

	$ test_tomita.py

The necessary steps:
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

