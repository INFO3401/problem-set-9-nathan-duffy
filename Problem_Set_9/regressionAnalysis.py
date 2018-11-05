from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import csv
import numpy as np
import pandas as pd

class analysisData:


    def __init__(self, filename):
        self.dataset = []
        self.variables = []
        self.filename = filename



    def parseFile(self):
        with open(self.filename, 'r') as f:
            reader = csv.reader(f, delimiter = ',')
            next(reader)

            for row in reader:
                self.dataset.append(row)

dataParser = analysisData("./candy-data.csv")
dataParser.parseFile()
print(dataParser.dataset)

class linearAnalysis(object):
    def __init__(self, targetY):
        self.bestX = None
        self.targetY = targetY
        self.fit = None

    def runSimpleAnalysis(self, dataParser):
        dataset = dataParser.dataset

        best_pred = 0
        for column in dataParser.variables:

            if column == self.targetY or column == 'competitorname':
                continue
            x_values = dataset[column].values.reshape(-1,1)
            y_values = dataset[self.targetY].x_values

            regr = linearRegression()
            regr.fit(x_values, y_values)
            preds = regr.predict(x_values)
            score = r2_score(y_values, preds)

            if score > best_pred:
                best_pred = r2_score
                self.bestX = column

        self.fit = best_pred
        print(self.bestX)
        print(self.fit)

linear_analysis = linearAnalysis(targetY = 'sugarpercent')
linear_analysis.runSimpleAnalysis(dataParser)
