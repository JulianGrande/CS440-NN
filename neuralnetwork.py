import numpy as np
import time
import util
class NeuralNetwork:

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "nn"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()

    def input_layer(self, trainingData, trainingLabels, validationData, validationLabels):
        print(len(trainingData))
        print(len(trainingLabels))
        #print(len(validationData))
        print(len(validationLabels))

        w_i_h = np.random.uniform(-0.5, 0.5, (20, 784)) # weights that brings 784 to 20 nodes
        w_h_o = np.random.uniform(-0.5, 0.5, (10, 20)) # weights that brings 20 nodes to 10 nodes to compare
        b_i_h = np.zeros((20, 1))
        b_h_o = np.zeros((10, 1))

        learn_rate = 0.01
        time_limit = 210

        total_samples = len(images)
        start_time = time.time()

        while True:
            for percentage in range(10, 101, 10):
                if time.time() - start_time >= time_limit:
                    print("Training terminated due to time limit.")
                    return  # Exit the loop if time is up

                subset_start_time = time.time()  # Record time for this subset
                subset_size = int(total_samples * (percentage / 100.0))
                training_subset = random.sample(list(zip(trainingData, trainingLabels)), subset_size)
                subset_data = [x[0] for x in training_subset]
                subset_labels = [x[1] for x in training_subset]
                nr_correct = 0
                for _ in range(self.max_iterations):  # Iterate over the subset
                    for datum, label in zip(subset_data, subset_labels):
                        datum.shape += (1,)
                        label.shape += (1,)

                        # Forward propagation input -> hidden
                        h_pre = b_i_h + w_i_h @ img
                        h = 1 / (1 + np.exp(-h_pre))

                        # Forward propagation hidden -> output
                        o_pre = b_h_o + w_h_o @ h
                        o = 1 / (1 + np.exp(-o_pre))

                        # Cost / Error calculation
                        e = 1 / len(o) * np.sum((o - l) ** 2)

                        # Backpropagation output -> hidden (cost function derivative)
                        delta_o = o - l
                        w_h_o += -learn_rate * delta_o @ h.T
                        b_h_o += -learn_rate * delta_o

                        # Backpropagation hidden -> input
                        delta_h = w_h_o.T @ delta_o * (h * (1 - h))
                        w_i_h += -learn_rate * delta_h @ img.T
                        b_i_h += -learn_rate * delta_h

                        # Accuracy calculation
                        nr_correct += int(np.argmax(o) == np.argmax(l))



        # for epoch in range(epoch):
        #     for img,label in zip(trainingData,trainingLabels):
        #         pixel_vector = np.array(list(img.values()))
        #         label_vector = np.zeros(10)
        #         label_vector[label] = 1
#
        #         print(pixel_vector * weights[counter])
        #         counter += 1
#
#

