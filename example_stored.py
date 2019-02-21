"""
This module is only used to show some example usage of the framework.
"""
from pypuf import tools
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


def main():
    """
    Run an example how to use pypuf.
    Developers Notice: Changes here need to be mirrored to README!
    """
    n = 128    # challenge bits
    k = 1      # k-xor PUF
    N = 12000  # total number of CRPs
    T = 10000  # number of CRPs to use for testing
    filename = 'test/data/fpga-puf-crps.txt'

    # read pairs from file
    training_set = tools.transform_challenge_01_to_11(tools.parse_file(filename, n, 1, N - T))
    testing_set = tools.transform_challenge_01_to_11(tools.parse_file(filename, n, N - T, T))

    # create the learner
    lr_learner = LogisticRegression(
        t_set=training_set,
        n=n,
        k=k,
        transformation=LTFArray.transform_atf,
        combiner=LTFArray.combiner_xor,
    )

    # learn and test the model
    model = lr_learner.learn()
    accuracy = 1 - tools.approx_dist_nonrandom(model, testing_set)

    # output the result
    print('Learned a {}-bit {}-xor XOR Arbiter PUF from {} CRPs with accuracy {}'.format(n, k, N, accuracy))


if __name__ == '__main__':
    main()
