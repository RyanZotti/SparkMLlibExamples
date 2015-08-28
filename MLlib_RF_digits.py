from pyspark.mllib.tree import RandomForest, RandomForestModel
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

model = RandomForest.trainClassifier(train_parsed, numClasses=10, categoricalFeaturesInfo={},
                                     numTrees=300, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

#predictions = model.predict(test_parsed.map(lambda x: x.features))
predictions = model.predict(test_parsed.map(lambda x: x.features))
labelsAndPredictions = test_parsed.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda t: t[0] != t[1]).count() / float(test_parsed.count())
print(testErr)
