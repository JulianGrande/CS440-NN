# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation

import util
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
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
        
        percentages = [20, 30, 40, 50, 60, 70, 80, 90, 100]  # Define the percentages
        
        for percent in percentages:
            # Determine the subset of training data based on the percentage
            subset_size = int(len(trainingData) * (percent / 100.0))
            subset_data = trainingData[:subset_size]
            subset_labels = trainingLabels[:subset_size]

            print(f"\nTraining with {percent}% of the data ({subset_size} samples)")
            
            # Reset weights for each iteration of training with different sizes
            self.weights = {label: util.Counter() for label in self.legalLabels}
            
            # Train the perceptron with the specified data subset
            for iteration in range(self.max_iterations):
                for datum, label in zip(subset_data, subset_labels):
                    # Classify the datum
                    scores = util.Counter()
                    for l in self.legalLabels:
                        scores[l] = self.weights[l] * datum
                    guessedLabel = scores.argMax()

                    # Update weights if the guess was incorrect
                    if guessedLabel != label:
                        self.weights[label] += datum  # Strengthen correct weights
                        self.weights[guessedLabel] -= datum  # Weaken incorrect weights

            # Validate and print accuracy for this subset
            guesses = self.classify(validationData)
            correct_count = sum([guesses[i] == validationLabels[i] for i in range(len(validationLabels))])
            accuracy = (correct_count / len(validationLabels)) * 100
            print(f"Validation accuracy with {percent}% training data: {accuracy:.2f}%")

    def classify(self, data):
        """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
            return guesses


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