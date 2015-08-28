from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, LogisticRegressionModel
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

# Function to convert .csv files into 'LabeledPoint' format
def parsePoint(line):
    values = [float(x.strip()) for x in line.split(',')]
    return LabeledPoint(values[-1],values[:65])

# Load .csv data
train_csv = sc.textFile("train.csv")
test_csv = sc.textFile("test.csv")

# Convert the data to LabeledPoint format
train_parsed = train_csv.map(parsePoint)
test_parsed = test_csv.map(parsePoint)

# Traing the model
model = LogisticRegressionWithLBFGS.train(train_parsed, iterations=10,numClasses=10)
#model = LogisticRegressionWithSGD.train(train_parsed, iterations=10, numClasses=10)
#model = SVMWithSGD.train(train_parsed,iterations=10)

# Get predictions and see how it did
predictions = model.predict(test_parsed.map(lambda x: x.features))
labelsAndPredictions = test_parsed.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda t: t[0] != t[1]).count() / float(test_parsed.count())
print(testErr)
