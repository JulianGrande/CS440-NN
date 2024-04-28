import numpy as np
import util
class NeuralNetwork:

    def input_layer(trainingData, trainingLabels, validationData, validationLabels):
        print(len(trainingData))
        print(len(trainingLabels))
        #print(len(validationData))
        print(len(validationLabels))

        weights = np.random.uniform(-0.5, 0.5, (10, 784))
        biased = np.random.uniform(-0.5, 0.5, (10, 1))


        epoch = 1
        counter = 0
        for epoch in range(epoch):
            for img,label in zip(trainingData,trainingLabels):
                pixel_vector = np.array(list(img.values()))
                label_vector = np.zeros(10)
                label_vector[label] = 1

                print(pixel_vector * weights[counter])
                counter += 1




