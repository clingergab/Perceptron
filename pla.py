import numpy as np
import pandas as pd
import sys
import copy
from plot_db import visualize_scatter

class pla:
    weights = []

    def __init__(self):
        #offset is weights[2]
        self.weights = [0, 0, 0]
        

    def cFunction(self, data):
        output = self.weights[2] + self.weights[0] * data[0] + self.weights[1] * data[1]
        return output

    def train(self, data):
        #for element in data:
        if data[2] * self.sign(self.cFunction(data)) <= 0:
            self.update(data)


    def update(self, data):
        for i in range(2):
            self.weights[i] += data[2] * data[i]
        self.weights[2] += data[2]

    def sign(self, input):
        if input > 0:
            return 1
        else:
            return -1

    def getWeights(self):
        return (self.weights[0], self.weights[1], self.weights[2])


if __name__ == "__main__":
    #print(np.__version__)
    csvFile = pd.read_csv(sys.argv[1], header=None)
    #print(csvFile)
    data = np.array(csvFile)
    f = open(sys.argv[2], "w")
    p = pla()
    unconverged = True
    while(unconverged):
        weightsCpy = copy.deepcopy(p.weights)
        for row in data:
            p.train(row)
            f.write(str(p.weights) + "\n")
        if weightsCpy == p.weights:
            unconverged = False
                
    f.close()

    visualize_scatter(df=csvFile, feat1=0, feat2=1, labels=2, weights=p.weights, title="perceptron")

