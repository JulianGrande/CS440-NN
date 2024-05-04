import numpy as np
import time
import util
import random
class NeuralNetwork:

    def input_layer(trainingData, trainingLabels, validationData, validationLabels,p_length):

        # input layer: 784 neurons
        # hidden layer: 20 neurons
        # output layer: 10 neurons









        w_i_h = np.random.uniform(-0.5, 0.5, (20, p_length))
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
            for img, label in zip(trainingData, trainingLabels):
                pixel_vector = np.array(list(img.values()))
                pixel_vector.shape +=(1,)

                label_vector = np.zeros(10)
                label_vector[label] = 1
                label_vector.shape +=(1,)

                # Forward Prop.
                h_pre = b_i_h + w_i_h @ pixel_vector
                h = 1 / (1 + np.exp(-h_pre))

                o_pre = b_h_o + w_h_o @ h
                o = 1 / (1 + np.exp(-o_pre))

                e = 1 / len(o) * np.sum((o - label) ** 2)
                nr_correct += int(np.argmax(o) == np.argmax(label))

                delta_o = o - label


                w_h_o += -learn_rate * delta_o @ h.T

                b_h_o += -learn_rate * delta_o


                delta_h = w_h_o.T @ delta_o * (h * (1 - h))
                w_i_h += -learn_rate * delta_h @ pixel_vector.T
                b_i_h += -learn_rate * delta_h











