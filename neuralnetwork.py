import numpy as np
import time
import util
import random
import time
import warnings
class NeuralNetwork:

    def input_layer(trainingData, trainingLabels, validationData, validationLabels,p_length,output_length):

        # input layer: 784 neurons
        # hidden layer: 20 neurons
        # output layer: 10 neurons
        warnings.filterwarnings('ignore')


        w_i_h = np.random.uniform(-0.5, 0.5, (20, p_length))
        w_h_o = np.random.uniform(-0.5, 0.5, (output_length, 20))

        # initialize biases

        b_i_h = np.zeros((20, 1))
        b_h_o = np.zeros((output_length, 1))



        learn_rate = 0.01

        nr_correct = 0
        total_samples = len(trainingData)
        start_time = int(time.time())
        time_limit = start_time + 5
        while time_limit > time.time():
            for img, label in zip(trainingData, trainingLabels):
                pixel_vector = np.array(list(img.values()))
                pixel_vector.shape +=(1,)

                label_vector = np.zeros(output_length)
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
        print('DONE')









