"""
This module is only used to show some example usage of the framework.
"""
import argparse
from pypuf import tools
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


def uint(val):
    """
    Assures that the passed integer is positive.
    """
    ival = int(val)
    if ival <= 0:
        raise argparse.ArgumentTypeError('{} is not a positive integer'.format(val))
    return ival

def main():
    """
    Run an example how to use pypuf.
    Developers Notice: Changes here need to be mirrored to README!
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=uint,
                        help='challenge bits')
    parser.add_argument('k', type=uint,
                        help='number of arbiter chains')
    parser.add_argument('num_tr', type=uint,
                        help='number of CRPs to use for training')
    parser.add_argument('num_te', type=uint,
                        help='number of CRPs to use for testing')
	parser.add_argument('combiner', type=str,
						help='used to combine the output bits to a single bit. Currently available: "ip_mod2", "xor", "majority_vote"')
    args = parser.parse_args()
    
	n = args.n
	k = args.k
	num_tr = args.num_tr
	num_te = args.num_te
	combiner = 'LTFArray.combiner_' + args.combiner
	
    # create a simulation with random (Gaussian) weights
    # for 64-bit 2-XOR
    instance = LTFArray(
        weight_array=LTFArray.normal_weights(n, k),
        transform=LTFArray.transform_atf,
        combiner=LTFArray.combiner_xor,
    )

    # create the learner
    lr_learner = LogisticRegression(
        t_set=tools.TrainingSet(instance=instance, N = num_tr),
        n = n,
        k = k,
        transformation=LTFArray.transform_atf,
        combiner=LTFArray.combiner_xor,
    )

    # learn and test the model
    model = lr_learner.learn()
    accuracy = 1 - tools.approx_dist(instance, model, args.num_te)

    # output the result
    print(accuracy)

if __name__ == '__main__':
    main()
