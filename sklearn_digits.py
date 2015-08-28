from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn.ensemble import GradientBoostingClassifier as GBM
from sklearn.ensemble import RandomForestClassifier as RF
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
digits = load_digits()
predictors = pd.DataFrame(digits['data'])
target = pd.DataFrame(digits['target'])
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.33, random_state=42)

# Gradient Boosting model
tune_parameters = [{'n_estimators':[10,100,150,200,300],
                    'learning_rate':[0.1,0.01],
                    'max_depth':[1,2,3]}]
gbm = grid_search.GridSearchCV(GBM(), tune_parameters,n_jobs=8).fit(X_train,np.ravel(y_train))
predictions = gbm.predict(X_test)
accuracy_score(y_test, predictions)

# Random Forest model
tune_parameters = [{'n_estimators':[10,100,150,200,300],'max_depth':[4]}]
model = grid_search.GridSearchCV(RF(), tune_parameters,n_jobs=8).fit(X_train,np.ravel(y_train))
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)

# Write data to files so that MLlib can view them
df = pd.DataFrame(digits['data'])
df['target'] = digits['target']
train, test, useless1, useless2 = train_test_split(df, df, test_size=0.33, random_state=42)
train.to_csv(path_or_buf='train.csv',header=False)
test.to_csv(path_or_buf='test.csv',header=False)
