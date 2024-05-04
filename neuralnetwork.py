import numpy as np
import time
import util
import random
import time
import warnings
from time import sleep
class NeuralNetwork:

    def input_layer(trainingData, trainingLabels, validationLabels,validationData,testLabels, testData,p_length,output_length):

        # input layer: 784 neurons
        # hidden layer: 20 neurons
        # output layer: 10 neurons


        validationData_len = len(validationData)
        warnings.filterwarnings('ignore')


        hidden_layer_len = 20
        w_i_h = np.random.uniform(-0.5, 0.5, (hidden_layer_len, p_length))
        w_h_o = np.random.uniform(-0.5, 0.5, (output_length, hidden_layer_len))

        # initialize biases

        b_i_h = np.zeros((hidden_layer_len, 1))
        b_h_o = np.zeros((output_length, 1))

        nr_testing = len(trainingLabels)


        learn_rate = 0.01

        nr_correct = 0
        num_of_loops = 1
        start_time = int(time.time())
        time_limit = start_time + 100
        while time_limit > time.time():
            for img, label in zip(trainingData, trainingLabels):
                pixel_vector = np.array(list(img.values()))
                pixel_vector.shape +=(1,)

                label_vector = np.zeros(output_length)
                label_vector[label] = 1
                label_vector.shape += (1,)

                # Forward Prop.
                h_pre = b_i_h + w_i_h @ pixel_vector
                h = 1 / (1 + np.exp(-h_pre))

                o_pre = b_h_o + w_h_o @ h
                o = 1 / (1 + np.exp(-o_pre))


                nr_correct += int(np.argmax(o) == np.argmax(label_vector))


                delta_o = o - label_vector


                w_h_o += -learn_rate * delta_o @ h.T

                b_h_o += -learn_rate * delta_o


                delta_h = w_h_o.T @ delta_o * (h * (1 - h))




                w_i_h += -learn_rate * delta_h @ pixel_vector.T
                b_i_h += -learn_rate * delta_h



            if nr_correct == nr_testing:
                print("It took",num_of_loops,'loops to reach 100% acc')
                break
            num_of_loops += 1

            nr_correct = 0

        print("Validating...")
        # Classify data
        count_correct = 0
        for data, labels in zip(validationData, validationLabels):
            data_vector = np.array(list(data.values()))
            data_vector.shape += (1,)


            h_pre = b_i_h + w_i_h @ data_vector
            h = 1 / (1 + np.exp(-h_pre))

            o_pre = b_h_o + w_h_o @ h
            o = 1 / (1 + np.exp(-o_pre))

            o_num = np.argmax(o)
            if o_num == labels:
                count_correct+= 1
        print(count_correct / validationData_len)

        print("Testing...")
        # Test data
        count_correct = 0
        for data, labels in zip(testData, testLabels):
            data_vector = np.array(list(data.values()))
            data_vector.shape += (1,)

            h_pre = b_i_h + w_i_h @ data_vector
            h = 1 / (1 + np.exp(-h_pre))

            o_pre = b_h_o + w_h_o @ h
            o = 1 / (1 + np.exp(-o_pre))

            o_num = np.argmax(o)
            if o_num == labels:
                count_correct += 1
        print(count_correct / validationData_len)









