# -*- coding: utf-8 -*-
from pypuf import simulation, learner, tools
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.learner.regression.logistic_regression import LogisticRegression
import pypuf.tools
from  LightweightMetaLearner import LightweightMetaLearner

import numpy as np
from numpy.random import RandomState
import datetime




numXors = 5
numStages = 64
numAttackedInstances = 200
trainingSetSize = 100000

filename = ('experimentResults_stages_' + str(numStages) + '_xors_' + str(numXors)
            + '_trainSetSize_' + str(trainingSetSize) +  '_time_{:%Y_%m_%d__%H_%M_%S}'.format(datetime.datetime.now()) )
print(filename)



results = np.zeros((numAttackedInstances, 5))
with open(filename, 'wb') as f:
    np.save(f, results)

for iteration in range(numAttackedInstances):
    print('+++++++++++++++ Running Experiment Iteration #' + str(iteration) + ' ++++++++++++++++-')
    instance = LTFArray(
        weight_array=LTFArray.normal_weights(n=64, k=4, random_instance=RandomState(seed=0xc0ffee)),
        transform=LTFArray.transform_lightweight_secure_original,
        combiner=LTFArray.combiner_xor,
        bias=0.0,
    )

    #If you want to actually simulate the attack set skipActualOptimizeLearning to False!
    meta = LightweightMetaLearner(
        instance,
        training_set=tools.TrainingSet(instance=instance, N=trainingSetSize, random_instance=RandomState(seed=0xdead)),
        validation_set=tools.TrainingSet(instance=instance, N=5000, random_instance=RandomState(seed=0xbeef)),
        maxNumberOptimizingTrials=-1,
        skipActualOptimizeLearning = False,
    )
    #numOfInitialTrials, numOfOptiTrials, initialModelAccuracy, optimizedModelAccuracy
    results[iteration, :] =  meta.learn()
    print('--- ' + str(results[iteration, 4]))
    with open(filename, 'wb') as f:
        np.save(f, results)

print(results)