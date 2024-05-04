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

        print(len(trainingData))
        print(len(w_i_h))


        learn_rate = 0.01
        time_limit = 210

        total_samples = len(trainingData)
        start_time = time.time()

        iterations = 10

        epochs = 3
        for epoch in range(epochs):
            for img, label in zip(trainingData, trainingLabels):
                pixel_vector = np.array(list(img.values()))

                label_vector = np.zeros(10)
                label_vector[label] = 1

                # Forward Prop.
                h_pre = b_i_h + w_i_h @ pixel_vector
                h = 1 / (1 + np.exp(-h_pre))
                print(h)




       # while True:
       #     for percentage in range(10, 101, 10):
       #         if time.time() - start_time >= time_limit:
       #             print("Training terminated due to time limit.")
       #             return  # Exit the loop if time is up
#
       #         subset_start_time = time.time()  # Record time for this subset
       #         subset_size = int(total_samples * (percentage / 100.0))
       #         training_subset = random.sample(list(zip(trainingData, trainingLabels)), subset_size)
       #         subset_data = [x[0] for x in training_subset]
       #         subset_labels = [x[1] for x in training_subset]
       #         nr_correct = 0
       #         for _ in range(iterations):  # Iterate over the subset
       #             for datum, label in zip(subset_data, subset_labels):
#
       #                 # forward propagation -> input to hidden layer
       #                 # h_pre = bias_inputlayer_to_hiddenlayer + dot_product of input_layer_weights, input_data
       #                 new_wih = w_i_h.T
       #                 h_pre = b_i_h + np.dot(subset_data, new_wih)
       #                 h = 1 / (1 + np.exp(-h_pre))
       #                 # subset_data: (500,)
       #                 # w_i_h: (20, 784)
#
       #                 # forward propagation -> hidden to output
       #                 o_pre = b_h_o + np.dot(subset_data, w_h_o)
       #                 o = 1 / (1 + np.exp(-o_pre))
#
       #                 # Cost / Error calculation
       #                 e = 1 / len(o) * np.sum((o - label) ** 2)
#
       #                 # Backpropagation output -> hidden (cost function derivative)
       #                 delta_o = o - label
       #                 w_h_o += -learn_rate * delta_o @ h.T
       #                 b_h_o += -learn_rate * delta_o
#
       #                 # Backpropagation hidden -> input
       #                 delta_h = w_h_o.T @ delta_o * (h * (1 - h))
       #                 w_i_h += -learn_rate * delta_h @ img.T
       #                 b_i_h += -learn_rate * delta_h
#
       #                 # Accuracy calculation
       #                 nr_correct += int(np.argmax(o) == np.argmax(label))




