# -*- coding: utf-8 -*-
# import classifip
# from classifip.models import ordinalLogit 
# from sklearn import cross_validation, datasets, metrics
import numpy as np
from classifip import dataset
from binarytree import dichotomies
# test = classifip.dataset.arff.ArffFile()



arff = dataset.arff.ArffFile()
arff.load("..//datasets//car.arff")
 
costs = dataset.genCosts.costMatrix(arff)
costs.ordinalCost()

nd = dichotomies.dichotomies(arff)

tree = nd.build_ordinal(arff)

tree.learnAll(dataset)


 

