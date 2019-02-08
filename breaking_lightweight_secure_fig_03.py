"""
Success Rate for Logistic Regression on 64-Bit, 4-XOR Arbiter PUF

Success rate of logistic regression attacks on simulated XOR Arbiter
PUFs with 64-bit arbiter chains and four arbiter chains each, based on at least
250 samples per data point shown. Accuracies better than 70% are considered
success (but cf. Figure 4). Four different designs are shown: of the four arbiter
chains in each instance, an input transform is used that transforms zero, one,
two, and three challenges pseudorandomly, keeping the remaining challenges
unmodified. The success rate decreases when the number of arbiter chains with
pseudorandom challenges is increased. The case with 4 pseudorandom sub-
challenges is not shown as it coincides with the results for 3 pseudorandom
challenges. Note the log-scale on the x-self.axisis.
"""
from pypuf.experiments.experimenter import Experimenter
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, CompoundTransformation
from pypuf.plots import SuccessRatePlot
from numpy.random import RandomState


samples_per_point = 10
success_threshold = .6

input_transformations = [
    LTFArray.transform_atf,
    CompoundTransformation(LTFArray.generate_stacked_transform, (LTFArray.transform_random, 1, LTFArray.transform_atf)),
    CompoundTransformation(LTFArray.generate_stacked_transform, (LTFArray.transform_random, 2, LTFArray.transform_atf)),
    CompoundTransformation(LTFArray.generate_stacked_transform, (LTFArray.transform_random, 3, LTFArray.transform_atf)),
]

training_set_sizes = [
    [
        1000,
        2000,
        5000,
        10000,
        12000,
        15000,
        20000,
        30000,
        50000,
        100000,
        200000,
        1000000,
    ],
    [
        1000,
        2000,
        5000,
        10000,
        20000,
        30000,
        40000,
        50000,
        100000,
        200000,
        300000,
        1000000,
    ],
    [
        2000,
        5000,
        10000,
        20000,
        40000,
        50000,
        60000,
        100000,
        200000,
        400000,
        600000,
        1000000,
    ],
    [
        2000,
        5000,
        20000,
        40000,
        50000,
        60000,
        80000,
        100000,
        200000,
        400000,
        600000,
        800000,
        1000000,
    ],
]


experiments = []
log = 'breaking_lightweight_secure_fig_03'
e = Experimenter(log, experiments)

for id, transformation in enumerate(input_transformations):
    for training_set_size in training_set_sizes[id]:
        for i in range(samples_per_point):
            experiments.append(
                ExperimentLogisticRegression(
                    log_name=log,
                    n=64,
                    k=4,
                    N=training_set_size,
                    seed_instance=314159 + i,
                    seed_model=265358 + i,
                    transformation=transformation,
                    combiner=LTFArray.combiner_xor,
                    seed_challenge=979323 + i,
                    seed_chl_distance=846264 + i,

                )
            )
RandomState(seed=1).shuffle(experiments)

result_plot = SuccessRatePlot('figures/breaking_lightweight_secure_fig_03.pdf', e.results)


def update_plot():
    result_plot.plot()


e.update_callback = update_plot
e.run()

result_plot.plot()
