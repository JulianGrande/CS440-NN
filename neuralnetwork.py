import numpy as np
import util
class NeuralNetwork:

    def input_layer(trainingData, trainingLabels, validationData, validationLabels):
        print(len(trainingData))
        print(len(trainingLabels))
        #print(len(validationData))
        print(len(validationLabels))

        weights = np.random.uniform(-0.5, 0.5, (10, 784))


        count = 0
        epoch = 1
        for epoch in range(epoch):
            for img in trainingData:
                pixel_vector = np.array(dict(img).values())
                print(pixel_vector)



        #for label in trainingLabels:
        #    vector = np.zeros(10)
        #    vector[label] = 1

            #print(vector)



        #print(trainingData)