# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation

import util
import numpy as np
import time
import random
PRINT = True

class PerceptronClassifier:
    """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter() # ds to use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights == weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the perceptron with incremental subsets of training data.
        The function loops through the subsets (20%, 30%, ..., 100%) and
        continues until the specified time limit is reached.
        """
        total_samples = len(trainingData)
        time_limit = 210  # 3.5 minutes in seconds
        start_time = time.time()  # Record the start time

        while True:
            # Loop through the percentages: 20%, 30%, ..., 100%
            for percentage in range(10, 101, 10):
                if time.time() - start_time >= time_limit:
                    print("Training terminated due to time limit.")
                    return  # Exit the loop if time is up

                subset_start_time = time.time()  # Record time for this subset
                subset_size = int(total_samples * (percentage / 100.0))
                training_subset = random.sample(list(zip(trainingData, trainingLabels)), subset_size)
                subset_data = [x[0] for x in training_subset]
                subset_labels = [x[1] for x in training_subset]

                # Train with the current subset
                for _ in range(self.max_iterations):  # Iterate over the subset
                    for datum, label in zip(subset_data, subset_labels):
                        predicted = self.classify_single(datum)
                        if predicted != label:
                            # Update the weights: Add the feature set for the correct label,
                            # subtract for the incorrect label
                            self.weights[label] += datum
                            self.weights[predicted] -= datum

                subset_end_time = time.time()  # End time for this subset
                subset_duration = subset_end_time - subset_start_time
                print(f"Training time for {percentage}% of data: {subset_duration:.2f} seconds")

        print("Training complete.")

    def classify(self, data):
        """
        Classify a list of data points and return the list of guessed labels.
        """
        guesses = []
        for datum in data:
            predicted = self.classify_single(datum)
            guesses.append(predicted)
        return guesses

    def classify_single(self, datum):
        """
        Classify a single datum as the label with the highest dot product with its weight vector.
        """
        vectors = util.Counter()
        for label in self.legalLabels:
            vectors[label] = self.weights[label] * datum  # Dot product
        return vectors.argMaxP()

    def findHighWeightFeatures(self, label):
        """
    Returns a list of the 100 features with the greatest weight for some label
    """
        featuresWeights = []

        if label in self.weights:
            labelWeights = self.weights[label]
            sortedFeatures = sorted(labelWeights.items(), key = lambda item: item[1], reverse = True)
            featuresWeights = sortedFeatures[:100]
        featuresWeights = [feature for feature, weight in featuresWeights]
        return featuresWeights