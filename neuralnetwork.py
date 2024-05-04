import numpy as np
import time
import util
import random
class NeuralNetwork:

    def input_layer(trainingData, trainingLabels, validationData, validationLabels):

        # input layer: 784 neurons
        # hidden layer: 20 neurons
        # output layer: 10 neurons

        # initialize weights







        w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
        w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))

        # initialize biases

        b_i_h = np.zeros((20, 1))
        b_h_o = np.zeros((10, 1))



        learn_rate = 0.01
        time_limit = 210
        nr_correct = 0
        total_samples = len(trainingData)
        start_time = time.time()

        iterations = 10

        epochs = 3
        for epoch in range(epochs):
            nr_correct = 0
            total_error = 0

            for img, label in zip(trainingData, trainingLabels):
                pixel_vector = np.array(list(img.values())).reshape(-1, 1)

                label_vector = np.zeros((10, 1))
                label_vector[label] = 1

                # Forward Propagation
                h_pre = b_i_h + w_i_h @ pixel_vector
                h = 1 / (1 + np.exp(-h_pre))

                o_pre = b_h_o + w_h_o @ h
                o = 1 / (1 + np.exp(-o_pre))

                # Error and Accuracy Tracking
                total_error += np.sum((o - label_vector) ** 2) / len(o)
                nr_correct += int(np.argmax(o) == label)

                # Backpropagation
                delta_o = o - label_vector
                w_h_o -= learn_rate * (delta_o @ h.T)
                b_h_o -= learn_rate * delta_o

                delta_h = (w_h_o.T @ delta_o) * (h * (1 - h))
                w_i_h -= learn_rate * (delta_h @ pixel_vector.T)
                b_i_h -= learn_rate * delta_h

            # Print accuracy and average error for the epoch
            accuracy = nr_correct / len(trainingData)
            average_error = total_error / len(trainingData)
            print(f"Epoch {epoch + 1}: Accuracy: {accuracy * 100:.2f}%, Avg Error: {average_error:.4f}")











