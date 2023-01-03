##################################################
# Language learner using LSTM neurons
# test using the first 5 Tomita languages
# from  M. Tomita, "Learning of construction of finite automata from examples using hill-climbing", 1982
# (C) Andreas Gaiser, 2022
###################################################

from lstm_automata import *
import re

alphabet = Alphabet(["0", "1"])


def tomita_1():
    # 1*
    language = Language(
        "Tomita1", alphabet, lambda word: len(word) >= 1 and word.find("0") == -1
    )
    topology = NeuralNetTopology(
        dense_layers_before_lstm=[10], dense_layers_after_lstm=[10], lstm_layers=[10]
    )
    learner = LSTMAutomataLearner(language, topology)
    learner.train(
        name_of_saved_model="model_for_%s" % language.name(),
        all_words_of_max_length=5,
        number_random_words=500,
        max_length_random_words=100,
        epochs_complete=1000,
        epochs_random=10,
        max_global_iterations=1000,
    )
    for i in range(1, 18):
        print(
            "Correct cases for strings of length %i: %s"
            % (i, str(learner.compute_precision_of_length(i)))
        )


def tomita_2():
    # (10)*
    language = Language(
        "Tomita2",
        alphabet,
        lambda word: len(word) >= 2
        and len(word) % 2 == 0
        and len(word.replace("10", "")) == 0,
    )
    topology = NeuralNetTopology(
        dense_layers_before_lstm=[10], dense_layers_after_lstm=[10], lstm_layers=[10]
    )
    learner = LSTMAutomataLearner(language, topology)
    learner.train(
        name_of_saved_model="model_for_%s" % language.name(),
        all_words_of_max_length=5,
        number_random_words=500,
        max_length_random_words=100,
        epochs_complete=1000,
        epochs_random=10,
        max_global_iterations=1000,
    )
    for i in range(1, 18):
        print(
            "Correct cases for strings of length %i: %s"
            % (i, str(learner.compute_precision_of_length(i)))
        )


def tomita_3():
    # after an odd number of consecutive 1's: never accept an odd number of consecutive 0s
    def is_accepting(word):
        current_block = ""
        found_odd_one_block = False
        for letter in word:
            if not any(current_block) or current_block[-1] == letter:
                current_block += letter
            else:
                if current_block[0] == "1" and len(current_block) % 2 == 1:
                    found_odd_one_block = True
                if (
                    found_odd_one_block
                    and current_block[0] == "0"
                    and len(current_block) % 2 == 1
                ):
                    return False
                current_block = letter
        return True

    language = Language("Tomita3", alphabet, is_accepting)
    topology = NeuralNetTopology(
        dense_layers_before_lstm=[10], dense_layers_after_lstm=[10], lstm_layers=[10]
    )
    learner = LSTMAutomataLearner(language, topology)
    learner.train(
        name_of_saved_model="model_for_%s" % language.name(),
        all_words_of_max_length=5,
        number_random_words=5000,
        max_length_random_words=100,
        epochs_complete=1000,
        epochs_random=10,
        max_global_iterations=1000,
    )
    for i in range(1, 18):
        print(
            "Correct cases for strings of length %i: %s"
            % (i, str(learner.compute_precision_of_length(i)))
        )


def tomita_4():
    # no 3 consecutive 0's

    language = Language("Tomita4", alphabet, lambda word: word.find("000") == -1)
    topology = NeuralNetTopology(
        dense_layers_before_lstm=[10], dense_layers_after_lstm=[10], lstm_layers=[10]
    )
    learner = LSTMAutomataLearner(language, topology)
    learner.train(
        name_of_saved_model="model_for_%s" % language.name(),
        all_words_of_max_length=5,
        number_random_words=500,
        max_length_random_words=100,
        epochs_complete=1000,
        epochs_random=10,
        max_global_iterations=1000,
    )
    for i in range(1, 18):
        print(
            "Correct cases for strings of length %i: %s"
            % (i, str(learner.compute_precision_of_length(i)))
        )


def tomita_5():
    # even number of 0's and even number of 1's

    language = Language(
        "Tomita5",
        alphabet,
        lambda word: len(word) >= 2
        and len(word) % 2 == 0
        and word.count("0") % 2 == 0
        and word.count("1") % 2 == 0,
    )
    topology = NeuralNetTopology(
        dense_layers_before_lstm=[10, 10],
        dense_layers_after_lstm=[10, 10],
        lstm_layers=[16, 16],
    )
    learner = LSTMAutomataLearner(language, topology)
    learner.train(
        name_of_saved_model="model_for_%s" % language.name(),
        all_words_of_max_length=5,
        number_random_words=1000,
        max_length_random_words=100,
        epochs_complete=1000,
        epochs_random=20,
        max_global_iterations=1000,
    )
    for i in range(1, 18):
        print(
            "Correct cases for strings of length %i: %s"
            % (i, str(learner.compute_precision_of_length(i)))
        )


def tomita_6():
    # number of 0'2 - of 1's mod 3 == 0

    language = Language(
        "Tomita6",
        alphabet,
        lambda word: len(word) >= 1 and (word.count("0") - word.count("1")) % 3 == 0,
    )
    topology = NeuralNetTopology(
        dense_layers_before_lstm=[5],
        dense_layers_after_lstm=[5],
        lstm_layers=[5],
        #   dense_layers_before_lstm=[10, 10],
        #   dense_layers_after_lstm=[10, 10],
        #   lstm_layers=[16, 16],
    )
    learner = LSTMAutomataLearner(language, topology)
    learner.train(
        name_of_saved_model="model_for_%s" % language.name(),
        all_words_of_max_length=6,
        number_random_words=1000,
        max_length_random_words=100,
        epochs_complete=100,
        epochs_random=10,
        max_global_iterations=1000,
    )
    for i in range(1, 18):
        print(
            "Correct cases for strings of length %i: %s"
            % (i, str(learner.compute_precision_of_length(i)))
        )


def tomita_7():
    # 0*1*0*1*

    pattern = re.compile("0*1*0*1*$")

    language = Language(
        "Tomita7",
        alphabet,
        lambda word: len(word) >= 1 and (pattern.match(word) is not None),
    )
    topology = NeuralNetTopology(
        dense_layers_before_lstm=[10],
        dense_layers_after_lstm=[10],
        lstm_layers=[16, 16],
        #    dense_layers_before_lstm=[10, 10],
        #    dense_layers_after_lstm=[10, 10],
        #    lstm_layers=[16, 16],
    )
    learner = LSTMAutomataLearner(language, topology)
    learner.train(
        name_of_saved_model="model_for_%s" % language.name(),
        all_words_of_max_length=5,
        number_random_words=2000,
        max_length_random_words=100,
        epochs_complete=500,
        epochs_random=20,
        max_global_iterations=1000,
    )
    for i in range(1, 18):
        print(
            "Correct cases for strings of length %i: %s"
            % (i, str(learner.compute_precision_of_length(i)))
        )


import functools, math


def custom_language_1():
    def is_prime(n):
        if n <= 1:
            return False
        if n % 2 == 0:
            return n == 2
        max_div = math.floor(math.sqrt(n))
        for i in range(3, 1 + max_div, 2):
            if n % i == 0:
                return False
        return True

    language = Language(
        "CustomLanguage1", alphabet, lambda word: is_prime(int(word, 2))
    )
    topology = NeuralNetTopology(
        dense_layers_before_lstm=[125, 125],
        dense_layers_after_lstm=[125, 125],
        lstm_layers=[125, 125],
    )
    learner = LSTMAutomataLearner(language, topology)
    learner.train(
        name_of_saved_model="model_for_%s" % language.name(),
        all_words_of_max_length=10,
        number_random_words=5000,
        max_length_random_words=50,
        epochs_complete=1000,
        epochs_random=10,
        max_global_iterations=90000,
    )
    print(
        "Loss for strings of length 15: %s right of %s elements."
        % learner.compute_precision_of_length(15)
    )


if __name__ == "__main__":

    tomita_1()
    tomita_2()
    tomita_3()
    tomita_4()
    tomita_5()
    # tomita_6()
    tomita_7()
    # custom_language_1()
