from pyspark.context import SparkContext
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

# API docs
# https://spark.apache.org/docs/1.4.0/api/python/pyspark.mllib.html

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

# Build a GBM / TreeNet model
model = GradientBoostedTrees.trainClassifier(
	train_parsed, loss='leastSquaresError', o
	categoricalFeaturesInfo={},numIterations=300, 
	maxDepth=2,learningRate=0.1)

# Get predictions and see how it did
predictions = model.predict(test_parsed.map(lambda x: x.features))
labelsAndPredictions = test_parsed.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda t: t[0] != t[1]).count() / float(test_parsed.count())
print(testErr)
