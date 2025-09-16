import os
import pickle
import re
from collections import defaultdict
from random import Random
from typing import Literal

import numpy as np
from scipy.special import expit

MODAL_LEXICON = {"would", "wouldn't", "would've", "could", "couldn't", "could've", "might"}
NEGATIONS = {'not', 'never', 'no'}

FeatureMode = Literal['bare', 'bow', 'both']


class LogisticRegression():
    def __init__(self, feature_mode: FeatureMode = 'bare'):
        self.feature_mode = feature_mode
        self.class_dict = {}
        self.feature_dict = {}
        self.eta, self.n_epochs, self.batch_size, self.n_features, self.theta, self.filenames, self.classes, self.documents, self.n_documents, self.mode = None, None, None, None, None, None, None, None, None, None

    def tokenize(self, text):
        """
        Returns a list of tokens extracted from text.
        :param text: the text to tokenize
        :return: a list of tokens
        """

        return re.findall(r'[\w]+|[^\w\s]', text.lower())

    def make_dicts(self, train_set):
        """
        Given a training set, fills in self.class_dict (and self.feature_dict)
        Also sets the number of features self.n_features and initializes the
        parameter vector self.theta.
        :param train_set: the training set to use
        :return: None
        """

        class_counter = 0
        vocab = set()

        # Traverse the training folder structure
        for class_name in sorted(os.listdir(train_set)):
            class_path = os.path.join(train_set, class_name)
            if not os.path.isdir(class_path):
                continue

            if class_name not in self.class_dict:
                self.class_dict[class_name] = class_counter
                class_counter += 1

            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    words = self.tokenize(text)
                    vocab.update(words)

        self.feature_dict = {word: i for i, word in enumerate(sorted(vocab))}

        # Additional features:
        # 1. 'Had' count
        # 2. Modal count
        # 3. Suffix -ed count
        # 4. Normalized negation count
        num_extra_features = 4

        # Number of features (+1 for the bias term)
        self.n_features = len(self.feature_dict) + num_extra_features + 1 if self.feature_mode == 'both' else len(
            self.feature_dict) + 1 if self.feature_mode == 'bow' else num_extra_features + 1

        # Initialize model parameters as zeros
        self.theta = np.zeros(self.n_features)

    def load_data(self, data_set):
        """
        Loads a dataset. Specifically, returns a list of filenames, and dictionaries
        of classes and documents such that:
        classes[filename] = class of the document
        documents[filename] = feature vector for the document (use self.featurize)
        :param data_set: the data set to load
        :return: a list of filenames, a dictionary of classes, and a dictionary of documents
        """

        filenames = []
        classes = {}
        documents = {}

        # Traverse the dataset folder
        for class_name in sorted(os.listdir(data_set)):
            class_path = os.path.join(data_set, class_name)
            if not os.path.isdir(class_path):  # Skip non-folder files
                continue

            # Get class index
            class_index = self.class_dict[class_name]

            # Process all documents in the class subfolder
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    words = self.tokenize(text)
                    feature_vector = self.featurize(words)  # Generate features

                    filenames.append(filename)
                    classes[filename] = class_index
                    documents[filename] = feature_vector

        return filenames, classes, documents

    def featurize(self, document):
        """
        Given a document (as a list of words), returns a feature vector.
        Note that the last element of the vector, corresponding to the bias, is a
        "dummy feature" with value 1.
        :param document: the document to featurize
        :return: the feature vector for the document
        """

        # Initialize feature vector with zeros.
        # The vector contains bag-of-words features plus 4 extra features and a bias.
        feature_vector = np.zeros(self.n_features)

        # Bag-of-words: count occurrences of words in the document.
        if self.feature_mode == 'bow' or self.feature_mode == 'both':
            for word in document:
                if word in self.feature_dict:
                    feature_index = self.feature_dict[word]
                    feature_vector[feature_index] += 1

        if self.feature_mode != 'bow':
            base_index = len(self.feature_dict) if self.feature_mode == 'both' else 0

            doc_length = len(document)
            norm = doc_length if doc_length > 0 else 1

            # 1. 'Had' count
            had_count = sum(1 for word in document if word.lower() == "had")
            feature_vector[base_index] = had_count / norm

            # 2. Modal count
            mod_count = sum(1 for word in document if word.lower() in MODAL_LEXICON)
            feature_vector[base_index + 1] = mod_count / norm

            # 3. Suffix -ed count
            ed_count = sum(1 for word in document if word.lower()[-2:] == "ed")
            feature_vector[base_index + 2] = ed_count / norm

            # 4. Normalized negation count
            negation_count = sum(1 for word in document if word.lower() in NEGATIONS)
            feature_vector[base_index + 3] = negation_count / norm

        # 5. Bias term
        feature_vector[-1] = 1

        return feature_vector

    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1, grid_search=False):
        """
        Trains a logistic regression classifier on a training set.
        :param train_set: the training set to use
        :param batch_size: batch size
        :param n_epochs: number of epochs to train for
        :param eta: the learning rate
        :param grid_search: whether it is being run within grid search
        :return: the loss of the training
        """
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.eta = eta

        # Load training data if not loaded
        if self.mode != 'train':
            self.filenames, self.classes, self.documents = self.load_data(train_set)
            self.mode = 'train'

        self.n_documents = len(self.documents)

        total_loss = 0

        for epoch in range(self.n_epochs):
            # Shuffle data
            Random(epoch).shuffle(self.filenames)

            X_full = np.array([self.documents[fname] for fname in self.filenames])
            y_full = np.array([self.classes[fname] for fname in self.filenames])

            total_loss = 0

            # Create mini-batches
            for i in range(0, len(self.filenames), self.batch_size):
                X_batch = X_full[i:i + self.batch_size]
                y_batch = y_full[i:i + self.batch_size]

                # Compute predictions using the sigmoid function
                y_hat = expit(np.dot(X_batch, self.theta))

                # Update the cross-entropy loss
                total_loss += np.sum(-(y_batch * np.log(y_hat + 1e-10) + (1 - y_batch) * np.log(1 - y_hat + 1e-10)))

                # Compute gradient of cross-entropy loss
                gradient = (1 / len(y_batch)) * np.dot(X_batch.T, (y_hat - y_batch))

                # Update parameters using gradient descent
                self.theta -= self.eta * gradient

            avg_epoch_loss = total_loss / len(self.filenames)
            if not grid_search:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Average Loss: {avg_epoch_loss}")
        return total_loss / len(self.filenames)

    def test(self, test_set):
        """
        Tests the classifier on a development or test set.
        Returns a dictionary of filenames mapped to their correct and predicted
        classes such that:
        results[filename]['correct'] = correct class
        results[filename]['predicted'] = predicted class
        :param test_set: the test set to test on
        :return: the results which are the correct and predicted classes for each file
        """

        # Load the development/test data
        if self.mode != 'test':
            self.filenames, self.classes, self.documents = self.load_data(test_set)
            self.mode = 'test'

        results = {}

        for filename in self.filenames:
            feature_vector = self.documents[filename]  # Get feature vector
            probability = expit(np.dot(feature_vector, self.theta))  # Compute P(y=1|x)

            # Apply decision boundary at 0.5
            predicted_class = 1 if probability > 0.5 else 0

            # Store correct and predicted classes
            results[filename] = {
                "correct": self.classes[filename],
                "predicted": predicted_class
            }

        return results

    def evaluate(self, results):
        """
        Given results, calculates the following:\n
        Precision, Recall, F1 for each class.\n
        Accuracy overall.\n
        Also, prints evaluation metrics in readable format.
        :param results: the results from testing
        :return: None
        """

        # Initialize counts for each class
        class_metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
        total_correct = 0
        total_samples = len(results)

        # Compute TP, FP, FN for each class
        for filename, res in results.items():
            true_label = res["correct"]
            predicted_label = res["predicted"]

            if true_label == predicted_label:
                total_correct += 1
                class_metrics[true_label]["TP"] += 1
            else:
                class_metrics[true_label]["FN"] += 1
                class_metrics[predicted_label]["FP"] += 1

        # Compute precision, recall, F1-score for each class
        print("Evaluation Metrics:")
        print("-" * 40)
        for class_label, metrics in sorted(class_metrics.items()):
            TP, FP, FN = metrics["TP"], metrics["FP"], metrics["FN"]
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            print(f"Class {class_label}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-score:  {f1_score:.4f}")
            print("-" * 40)

        # Compute overall accuracy
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        print(f"Overall Accuracy: {accuracy:.4f}")

    def print_top_features(self, top_n=20):
        """
        Prints the top-N most predictive features for each class.
        Positive weights => more predictive of class 1 (subjunctive)
        Negative weights => more predictive of class 0 (indicative)
        :param top_n: the number of top features to print
        :return: None
        """

        # Build reverse mapping from index to feature name
        index_to_feature = {}

        if self.feature_mode in ('bow', 'both'):
            for word, idx in self.feature_dict.items():
                index_to_feature[idx] = word

        if self.feature_mode != 'bow':
            base_index = len(self.feature_dict) if self.feature_mode == 'both' else 0
            engineered_names = [
                "had_count_norm",
                "modal_count_norm",
                "ed_suffix_count_norm",
                "negation_count_norm"
            ]
            for i, name in enumerate(engineered_names):
                index_to_feature[base_index + i] = name

        index_to_feature[self.n_features - 1] = "<BIAS>"

        # Get sorted indices by weight
        sorted_indices = np.argsort(self.theta)

        print(f"\nTop {top_n} features for class 0 (indicative):")
        for idx in sorted_indices[:top_n]:
            print(f"{index_to_feature.get(idx, str(idx)):<25} {self.theta[idx]:.4f}")

        print(f"\nTop {top_n} features for class 1 (subjunctive):")
        for idx in sorted_indices[-top_n:][::-1]:
            print(f"{index_to_feature.get(idx, str(idx)):<25} {self.theta[idx]:.4f}")

    def grid_search(self, train_set, test_set, batch_sizes: list[int], epochs: list[int] | int,
                    learning_rates: list[float]):
        """
        Performs a grid-search over all possible combinations of hyperparameters.
        Returns the model with the highest accuracy.
        :param train_set: the training set to use
        :param test_set: the test set to use
        :param batch_sizes: a list of batch sizes to test
        :param epochs: a list of epochs to test or the max epoch to test for
        :param learning_rates: a list of learning rates to test
        :return: the list of parameters and their corresponding accuracies
        """

        best_params = None
        best_accuracy = 0.0
        params_list = []
        epochs = epochs if type(epochs) is list else [i for i in range(1, epochs + 1)]
        max_epoch = max(epochs)
        iterations = len(batch_sizes) * max_epoch * len(learning_rates)
        iteration = 0

        # Iterate over all combinations of hyperparameters
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                # Reset model parameters
                self.theta = np.zeros(self.n_features)
                print(f"\nTesting batch_size={batch_size}, eta={learning_rate}")

                for n_epoch in range(max_epoch):
                    iteration += 1

                    # Train model
                    avg_loss = self.train(train_set, batch_size=batch_size, eta=learning_rate, grid_search=True)

                    if n_epoch + 1 in epochs:
                        # Evaluate on test set
                        results = self.test(test_set)
                        accuracy = sum(1 for res in results.values() if res["correct"] == res["predicted"]) / len(
                            results)

                        print(
                            f"Epoch {n_epoch + 1}/{max_epoch}, Average Loss: {avg_loss:.4f}, {100 * (iteration / iterations):.2f}% Completed, Accuracy: {accuracy:.4f}")

                        # Update best parameters
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = (batch_size, learning_rate, n_epoch + 1)

                        params_list.append((accuracy, (batch_size, learning_rate, n_epoch + 1)))
                    else:
                        print(
                            f"Epoch {n_epoch + 1}/{max_epoch}, Average Loss: {avg_loss:.4f}, {100 * (iteration / iterations):.2f}% Completed")

        # print("\nBest Hyperparameters:")
        # print(f"Batch Size: {best_params[0]}, Learning Rate: {best_params[1]}, Epochs: {best_params[2]}")
        # print(f"Best Accuracy: {best_accuracy:.4f}")

        return params_list

    def manual(self):
        """
        Prompts the user to manually classify sentences in an interactive loop. The user
        enters a sentence, which is tokenized and converted into a feature vector. The
        method then calculates the probability of the sentence being classified into one
        of two possible classes (e.g., subjunctive or indicative) using the logistic
        regression model parameters. The method interacts with the user until they
        choose to exit.
        """

        while True:
            sentence = input("Enter a sentence to classify (or 'q' to quit): ")
            if sentence == 'q':
                break
            words = self.tokenize(sentence)
            feature_vector = self.featurize(words)
            probability = expit(np.dot(feature_vector, self.theta))
            predicted_class = 1 if probability > 0.5 else 0
            print(f"Predicted class: {'Subjunctive' if predicted_class else 'Indicative'}")
            print(f"Probability: {probability * 100 if predicted_class else (1 - probability) * 100:.2f}%")

    # Add these methods inside the LogisticRegression class
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk using pickle.
        :param filepath: the path to save the model to (e.g., "models/logreg.pkl")
        :return: None
        """

        directory = os.path.dirname(os.path.abspath(filepath))
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> "LogisticRegression":
        """
        Load a previously saved LogisticRegression model from disk.
        :param filepath: the path to a pickle file created by `save`
        :return: the restored model instance
        """

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model file found at: {filepath}")
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"File does not contain a {cls.__name__} model.")
        return obj


def grid_search(train="data/train",
                test="data/test",
                batch_sizes=[1, 4, 8],
                epochs=[5, 10, 15, 20, 25, 30],
                learning_rates=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                mode: FeatureMode = "bare"):
    """
    Performs a grid search to evaluate the performance of a logistic regression classifier
    using various combinations of batch sizes, epochs, and learning rates. The function
    initializes a logistic regression classifier, processes training data, and iterates
    through the specified hyperparameter combinations to determine their corresponding
    accuracy. The results are sorted and displayed in descending order of accuracy.

    :param train: Path to the training dataset (default: "data/train")
    :param test: Path to the testing dataset (default: "data/test")
    :param batch_sizes: List of batch sizes to evaluate during the grid search
        (default: [1, 4, 8])
    :param epochs: List of epoch values to evaluate during the grid search
        (default: [5, 10, 15, 20, 25, 30])
    :param learning_rates: List of learning rate values to evaluate during the grid
        search (default: [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    :param mode: Mode of feature generation for initializing the logistic regression
        classifier (default: "bare")
    :return: A tuple containing the best hyperparameters and their corresponding accuracy
    """
    print("Performing Grid Search...")
    print("-" * 40)

    classifier = LogisticRegression(mode)
    classifier.make_dicts(train)

    params_list = classifier.grid_search(train, test, batch_sizes, epochs, learning_rates)
    # Sort with custom key
    params_list.sort(
        key=lambda x: (-x[0], x[1][2], -x[1][0], x[1][1])
    )

    # ANSI escape codes for light gray background and reset
    BG_LIGHT = "\033[47m"
    RESET = "\033[0m"

    print("\nParameter Results:")
    print(f"{'BS':<5}{'LR':<12}{'EP':<5}{'AC (%)':<8}")
    print("-" * 30)

    for idx, (acc, (bs, lr, ep)) in enumerate(params_list):
        line = f"{bs:<5}{lr:<12}{ep:<5}{acc * 100:<8.2f}"
        if idx % 2 == 0:
            print(f"{BG_LIGHT}{line}{RESET}")
        else:
            print(line)
    best_params = params_list[0][1]
    best_accuracy = params_list[0][0] * 100
    print("\nBest Hyperparameters:")
    print(f"Batch Size: {best_params[0]}, Learning Rate: {best_params[1]}, Epochs: {best_params[2]}")
    print(f"Best Accuracy: {best_accuracy:.2f}%\n")
    print("-" * 80)
    print()

    return mode, best_params


def new_model(train="data/train", test="data/test", mode: FeatureMode = "bare", batch_size=1, n_epochs=20, eta=0.5):
    """
    Trains a new logistic regression model based on the provided training dataset,
    evaluates it on a test dataset, and optionally saves the trained model. The
    model is built using the provided configuration parameters.

    :param train: Path to the training dataset file.
    :param test: Path to the test dataset file.
    :param mode: Feature mode to be used for building the model.
    :param batch_size: Number of samples processed per batch during training.
    :return: An instance of the trained LogisticRegression model.
    """
    print(f"Training New Model ({batch_size}, {eta}, {n_epochs}) in {mode.upper()} mode...\n")
    print("-" * 40)

    classifier = LogisticRegression(mode)
    classifier.make_dicts(train)
    classifier.train(train, batch_size, n_epochs, eta)

    results = classifier.test(test)
    classifier.evaluate(results)

    classifier.print_top_features()

    pickle_save = input("\nWould you like to save the model? (y/n): ")
    if pickle_save.lower() == 'y' or pickle_save.lower() == 'yes':
        model_name = input("Enter the path to save the model to (or q for default): ")
        if model_name == "q":
            path = f"model/{mode}_{batch_size}-{n_epochs}-{str(eta).replace(".", ",")}.pkl"
            classifier.save(path)
        else:
            path = f"model/{model_name}.pkl"
            classifier.save(path)
        print(f"Model saved to: {path}")
    print("-" * 80)
    print()

    return classifier


def load_model(path: str):
    """
    Load a logistic regression model from a specified path.

    :param path: The filename (without extension) of the model to load,
        specifying the relative path from the "model/" directory.
    :return: The loaded LogisticRegression model instance.
    """
    print("Loading Model...")
    classifier = LogisticRegression.load("model/" + path + ".pkl")
    print(
        f"Model Loaded: ({classifier.batch_size}, {classifier.n_epochs}, {classifier.eta}) in {classifier.feature_mode.upper()} mode")
    print("-" * 80)
    print()

    return classifier


def analyze_model(classifier: LogisticRegression, test="data/test"):
    print("Analyzing Model...\n")
    print(
        f"Model Hyperparameters: ({classifier.batch_size}, {classifier.n_epochs}, {classifier.eta}) in {classifier.feature_mode.upper()} mode")
    print(f"Number of Features: {classifier.n_features}")
    print(f"Number of Documents: {classifier.n_documents}")

    print("\nEvaluating Model...\n")
    results = classifier.test(test)
    classifier.evaluate(results)

    classifier.print_top_features()

    print("-" * 80)
    print()


def demo(classifier: LogisticRegression):
    """
    Executes the `manual` method of a provided classifier.

    :param classifier: The classifier instance implementing a `manual` method.
    :return: None
    """
    classifier.manual()


if __name__ == '__main__':
    # Do a grid search to find the best model hyperparameters
    mode, best_hyperparams = grid_search(batch_sizes=[1, 2, 4, 8], epochs=[5, 10, 15, 20, 25, 30],
                                         learning_rates=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                                         mode="bare")
    # Train a new model using those hyperparameters
    model = new_model(mode=mode, batch_size=best_hyperparams[0], n_epochs=best_hyperparams[2],
                           eta=best_hyperparams[1])

    # Load two pretrained models to analyze and use
    loaded_bow_model = load_model("bow_8-5-0,05")
    loaded_both_model = load_model("both_8-5-0,5")

    # Analyze the brains of the three models
    analyze_model(model)
    analyze_model(loaded_bow_model)
    analyze_model(loaded_both_model)

    # Do a manual demonstration of the loaded model
    demo(loaded_both_model)
