##################################################
# Language learner using LSTM neurons
# (C) Andreas Gaiser, 2022
###################################################

import logging
import tqdm
import random
import sys
from dataclasses import dataclass, field

import tensorflow
import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import keras as K

logging.getLogger().setLevel(logging.DEBUG)
sys.set_int_max_str_digits(20000)


class Alphabet:
    """Helper functions for the used alphabet, most often ["0", "1"] is used as letters."""

    def __init__(self, letters):
        self._letters = sorted(list(set(letters)))
        self._letter_to_id = {}
        self._id_to_letter = {}
        counter = 0
        for letter in sorted(self._letters):
            self._letter_to_id[letter] = counter
            self._id_to_letter[counter] = letter
            counter += 1
        self._dim = counter

    def one_hot_vector_from_letter(self, letter):
        return tensorflow.one_hot(self._letter_to_id[letter], self._dim)

    def letter_to_id(self, letter):
        return self._letter_to_id[letter]

    def id_to_letter(self, id):
        return self._id_to_letter[id]

    def dim(self):
        return self._dim

    def get_letters(self):
        return self._letters[:]


class Language:
    def __init__(self, name: str, the_alphabet: Alphabet, belongs_to_predicate):
        self._alphabet = the_alphabet
        self._belongs_to = belongs_to_predicate
        self._name = name

    def name(self):
        return self._name

    def alphabet(self):
        return self._alphabet

    def is_accepting(self, word):
        return self._belongs_to(word)

    def enumerate_words(self, length):
        def enumerate_words(word_length):
            if word_length == 0:
                yield ""
            else:
                for small_word in enumerate_words(word_length - 1):
                    for letter in self._alphabet.get_letters():
                        yield small_word + letter

        for word in enumerate_words(length):
            yield word, self._belongs_to(word)


# TODO: sigmoids that approximate binary step, for further experiments

SIGMOID_SCALING = 1

# These activations approximate binary step function
def custom_sigmoid_1(x):
    return keras.backend.sigmoid(x)


def custom_sigmoid_5(x):
    return keras.backend.sigmoid(5 * x)


def custom_sigmoid_10(x):
    return keras.backend.sigmoid(10 * x)


def custom_sigmoid_20(x):
    return keras.backend.sigmoid(20 * x)


def custom_sigmoid_50(x):
    return keras.backend.sigmoid(20 * x)


def global_is_accepting(word):
    return word[0] == word[1] and word[2] == word[3]


@dataclass
class NeuralNetTopology:
    """Topology settings for the neural net used during learning."""

    dense_layers_before_lstm: list[int] = field(default_factory=list)
    lstm_layers: list[int] = field(default_factory=list)
    dense_layers_after_lstm: list[int] = field(default_factory=list)


class LSTMAutomataLearner:
    """
    Computes a recurrent neural net that attempts to learn a given language:
    given a word as sequence of its letters, output 1 or 0 at the
    end of processing, indicating whether it thinks the word is belonging to the learned language or not."""

    def letter_count(self):
        return self._language.alphabet().dim()

    def alphabet(self):
        return self._language.alphabet()

    def _create_model(self):
        def get_initializer():
            return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=42)

        model = K.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(None, self.letter_count())))
        for pre_dense_layer_size in self._topology.dense_layers_before_lstm:
            model.add(
                keras.layers.Dense(
                    pre_dense_layer_size,
                    activation="relu",
                    kernel_initializer=get_initializer(),
                    bias_initializer=get_initializer(),
                )
            )
        lstm_layers_count = len(self._topology.lstm_layers)
        if lstm_layers_count != 0:
            import tensorflow_addons as tfa

            for lstm_layer_index in range(lstm_layers_count):
                cell = tfa.rnn.PeepholeLSTMCell(
                    self._topology.lstm_layers[lstm_layer_index],
                    recurrent_activation=custom_sigmoid_1,
                )
                rnn = tf.keras.layers.RNN(
                    cell,
                    return_sequences=lstm_layer_index != lstm_layers_count - 1,
                )
                model.add(rnn)
        for post_dense_layer_size in self._topology.dense_layers_before_lstm:
            model.add(
                keras.layers.Dense(
                    post_dense_layer_size,
                    activation="relu",
                    kernel_initializer=get_initializer(),
                    bias_initializer=get_initializer(),
                )
            )
        model.add(
            keras.layers.Dense(
                2,
                activation="softmax",
                kernel_initializer=get_initializer(),
                bias_initializer=get_initializer(),
            )
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        self._model = model

    def train(
        self,
        name_of_saved_model,
        all_words_of_max_length,
        number_random_words,
        max_length_random_words,
        epochs_complete,
        epochs_random,
        max_global_iterations,
    ):
        """
        Trains an RNN model with test words from the given language. It uses
        (1) all words of length <= all_words_of_max_length and
        (2) number_random_words many words of length <= max_length_random_words
         for training.
        Both categories of words are groupted into batches of words having the same length.
        Number of training epochs in each iteration for both categories are given by epochs_complete
        and epochs_random;
        A test set of random words is also created (similar constraints as for (2));
        if for each batch the loss is < 0.0001, the training ends and True is returned;
        otherwise the training ends after max_global_iterations of training categories (1) and (2) in sequence.
        Returns True iff the training is considered successful. Always saves model at the end.
        """

        def generate_test(length):
            s = ""
            for _ in range(length):
                rand_letter = self.alphabet().id_to_letter(
                    random.randint(0, self.alphabet().dim() - 1)
                )
                s += rand_letter
            return s, self._language.is_accepting(s)

        def word_to_test(the_input, the_output, test_index, word, is_accepting):
            """Set the test_index-th input/output entry to the input/output given by word/is_accepting."""
            letter_index = 0
            for letter in word:
                the_input[test_index][letter_index][
                    self.alphabet().letter_to_id(letter)
                ] = 1.0
                letter_index += 1
            the_output[test_index][0 if is_accepting else 1] = 1.0

        def create_all_words_of_length_batches():
            # Create batches with all words of a certain length
            input_batches = []
            output_batches = []
            for length in tqdm.tqdm(range(1, all_words_of_max_length + 1)):
                word_accept_pairs = list(self._language.enumerate_words(length))
                input_instance = np.zeros(
                    (len(word_accept_pairs), length, self.letter_count())
                )
                output_instance = np.zeros((len(word_accept_pairs), 2))
                word_index = 0
                for (word, is_accepting) in word_accept_pairs:
                    word_to_test(
                        input_instance, output_instance, word_index, word, is_accepting
                    )
                    word_index += 1
                input_batches.append(input_instance)
                output_batches.append(output_instance)
            return input_batches, output_batches

        def create_random_batches():
            # Create batches with random words
            logging.info("Create random batches...")
            random_words = {}
            for _ in tqdm.tqdm(range(number_random_words)):
                word, is_accepting = generate_test(
                    random.randint(1, max_length_random_words)
                )
                random_words.setdefault(len(word), []).append((word, is_accepting))
            input_batches = []
            output_batches = []
            for length in tqdm.tqdm(sorted(random_words.keys())):
                entries = random_words[length]
                X = np.zeros((len(entries), length, self.letter_count()))
                Y = np.zeros((len(entries), 2))
                word_index = 0
                for (word, is_accepting) in entries:
                    word_to_test(X, Y, word_index, word, is_accepting)
                    word_index += 1
                input_batches.append(X)
                output_batches.append(Y)
            logging.info("...done creating random batches.")
            return input_batches, output_batches

        X_random_batches, Y_random_batches = create_random_batches()
        (
            X_all_words_of_length_batches,
            Y_all_words_of_length_batches,
        ) = create_all_words_of_length_batches()
        X_test_batches, Y_test_batches = create_random_batches()

        def train_all_words_of_length_batches():
            length_tests = list(
                zip(X_all_words_of_length_batches, Y_all_words_of_length_batches)
            )
            random.shuffle(length_tests)
            for (X_instance, Y_instance) in length_tests:
                logging.info(
                    "Learning all words of length: %s, training set size: %s"
                    % (X_instance[0].shape[0], X_instance.shape[0])
                )
                history = self._model.fit(
                    X_instance, Y_instance, epochs=epochs_complete, verbose=0
                )
                logging.info("....Last loss: %s" % history.history["loss"][-1])

        def train_random_batches():
            random_tests = list(zip(X_random_batches, Y_random_batches))
            random.shuffle(random_tests)
            for (X_instance, Y_instance) in tqdm.tqdm(random_tests):
                logging.info(
                    "Learning random words, length: %s, training set size: %s"
                    % (X_instance.shape[0], Y_instance.shape[0])
                )
                history = self._model.fit(
                    X_instance, Y_instance, epochs=epochs_random, verbose=0
                )
                logging.info("Last loss: %s" % history.history["loss"][-1])

        def is_precision_sufficient():
            """Tests loss on test set; saves model and returns True iff loss is sufficiently small."""
            logging.info("Test all random test cases to assess current loss.")
            random_tests = list(zip(X_test_batches, Y_test_batches))
            random.shuffle(random_tests)
            # Check whether maybe all batches are already satsifiably good approximateed
            scores_satisfied = True
            for (X_instance, Y_instance) in random_tests:
                logging.info(
                    "Testing random words, length: %s, size: %s"
                    % (X_instance.shape[0], Y_instance.shape[0])
                )
                score = self._model.evaluate(X_instance, Y_instance, verbose=0)
                logging.info("Test loss: %s", score[0])
                if score[0] <= 0.0001:
                    logging.info("Small loss for random sample...")
                else:
                    logging.info(
                        "Loss too large for considering this language as learned."
                    )
                    scores_satisfied = False
                    break
            if scores_satisfied:
                logging.info(
                    "Small losses for all random samples, exiting and saving model."
                )
                return True
            return False

        precision_sufficient = False
        # Main loop
        for i in tqdm.tqdm(range(max_global_iterations)):
            logging.info("Global iteration %s" % i)
            random_tests = list(zip(X_random_batches, Y_random_batches))
            random.shuffle(random_tests)
            for (input_instance, output_instance) in random_tests:
                if is_precision_sufficient():
                    precision_sufficient = True
                    break
                train_all_words_of_length_batches()
                train_random_batches()

            if precision_sufficient:
                logging.info("Considering training successful.")
                self._model.save("MODEL_SAVED_SUCCESFUL_%s" % name_of_saved_model)
                return True

        logging.info("Failed in training model.")
        self._model.save("MODEL_SAVED_FAIL_%s_%s" % name_of_saved_model)
        return False

    def is_considered_accepting(self, word):
        input_instance = np.zeros((1, len(word), self.letter_count()))
        letter_index = 0
        for letter in word:
            input_instance[0][letter_index][self.alphabet().letter_to_id(letter)] = 1.0
            letter_index += 1
        return np.argmax(self._model.predict(input_instance, verbose=0)[0]) == 0

    def compute_precision_of_length(self, length):
        word_accept_pairs = list(self._language.enumerate_words(length))
        X_instance = np.zeros((len(word_accept_pairs), length, self.letter_count()))
        accepting = []
        word_index = 0
        for (word, is_accepting) in word_accept_pairs:
            letter_index = 0
            for letter in word:
                X_instance[word_index][letter_index][
                    self.alphabet().letter_to_id(letter)
                ] = 1.0
                letter_index += 1
            accepting.append(is_accepting)
            word_index += 1
        result = self._model.predict(X_instance, verbose=0)
        correct_cases = 0
        for i in range(len(accepting)):
            if (np.argmax(result[i]) == 0) == accepting[i]:
                correct_cases += 1
        return correct_cases, len(accepting)

    def compute_precision(self, length):
        word_accept_pairs = list(self._language.enumerate_words(length))
        correct_cases = 0
        all_cases = 0
        for (word, is_accepting) in tqdm.tqdm(word_accept_pairs):
            all_cases += 1
            if is_accepting == self.is_considered_accepting(word):
                correct_cases += 1
        return correct_cases, all_cases

    def __init__(self, language: Language, topology: NeuralNetTopology):
        self._topology = topology
        self._language = language
        self._create_model()
