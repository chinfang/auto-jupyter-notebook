import numpy as np
import openml as oml

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Silence warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

'''

Compute Landmarking meta-features according to Matthias Reif et al. 2012.
The accuracy values of the following simple learners are used: 
Naive Bayes, Linear Discriminant Analysis, One-Nearest Neighbor, 
Decision Node, Random Node.

'''

class LandmarkingMetafeatures():

    def __init__(self):
        pass

    def compute(self, dataset):
    	data = oml.datasets.get_dataset(dataset)
    	X, y = data.get_data(target=data.default_target_attribute)
    	self.landmarkers = get_landmarkers(X, y)


def pipeline(X, y, estimator):
    pipe = Pipeline([('Imputer', preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)),
                     ('classifiers', estimator)])
    score = np.mean(cross_val_score(pipe, X, y, cv=10, scoring='roc_auc', n_jobs=-1))
    return score

def get_landmarkers(X, y):
    landmarkers = {}
    landmarkers['one_nearest_neighbor'] = pipeline(X, y, KNeighborsClassifier(n_neighbors = 1)) 
    landmarkers['linear_discriminant_analysis'] = pipeline(X, y, LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')) 
    landmarkers['naive_bayes'] = pipeline(X, y, GaussianNB()) 
    landmarkers['decision_node'] = pipeline(X, y, DecisionTreeClassifier(criterion='entropy', splitter='best', 
                                                                         max_depth=1, random_state=0)) 
    landmarkers['random_node'] = pipeline(X, y, DecisionTreeClassifier(criterion='entropy', splitter='random',
                                                                       max_depth=1, random_state=0))
    return landmarkers


test = LandmarkingMetafeatures()
test.compute(1464)
print(test.landmarkers)