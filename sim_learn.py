from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from sys import argv, stdout, stderr
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experimenter import Experimenter
from random import sample
from string import ascii_uppercase


def main(args):

    if len(args) != 10:
        stderr.write('LTF Array Simulator and Logistic Regression Learner\n')
        stderr.write('Usage:\n')
        stderr.write('sim_learn.py n k transformation combiner N restarts seed_instance seed_model\n')
        stderr.write('               n: number of bits per Arbiter chain\n')
        stderr.write('               k: number of Arbiter chains\n')
        stderr.write('  transformation: used to transform input before it is used in LTFs\n')
        stderr.write('                  currently available:\n')
        stderr.write('                  - id  -- does nothing at all\n')
        stderr.write('                  - atf -- convert according to "natural" Arbiter chain\n')
        stderr.write('                           implementation\n')
        stderr.write('                  - mm  -- designed to achieve maximum PTF expansion length\n')
        stderr.write('                           only implemented for k=2 and even n\n')
        stderr.write('                  - lightweight_secure -- design by Majzoobi et al. 2008\n')
        stderr.write('                                          only implemented for even n\n')
        stderr.write('                  - 1_n_bent -- one LTF gets "bent" input, the others id\n')
        stderr.write('                  - 1_1_bent -- one bit gets "bent" input, the others id,\n')
        stderr.write('                                this is proven to have maximum PTF\n')
        stderr.write('                                length for the model\n')
        stderr.write('                  - polynomial -- challenges are interpreted as polynomials\n')
        stderr.write('                                  from GF(2^64). From the initial challenge c,\n')
        stderr.write('                                  the i-th Arbiter chain gets the coefficients \n')
        stderr.write('                                  of the polynomial c^(i+1) as challenge.\n')
        stderr.write('                                  For now only challenges with length n=64 are accepted.\n')
        stderr.write('        combiner: used to combine the output bits to a single bit\n')
        stderr.write('                  currently available:\n')
        stderr.write('                  - xor     -- output the parity of all output bits\n')
        stderr.write('                  - ip_mod2 -- output the inner product mod 2 of all output\n')
        stderr.write('                               bits (even n only)\n')
        stderr.write('               N: number of challenge response pairs in the training set\n')
        stderr.write('        restarts: number of repeated initializations the learner\n')
        stderr.write('                  use float number x, 0<x<1 to repeat until given accuracy\n')
        stderr.write('       instances: number of repeated initializations the instance\n')
        stderr.write('                  The number total learning attempts is restarts*instances.\n')
        stderr.write('   seed_instance: random seed used for LTF array instance\n')
        stderr.write('      seed_model: random seed used for the model in first learning attempt\n')
        quit(1)

    n = int(args[1])
    k = int(args[2])
    transformation_name = args[3]
    combiner_name = args[4]
    N = int(args[5])

    if float(args[6]) < 1:
        restarts = float('inf')
        convergence = float(args[6])
    else:
        restarts = int(args[6])
        convergence = 1.1

    instances = int(args[7])

    seed_instance = int(args[8], 16)
    seed_model = int(args[9], 16)

    log_file_prefix = ''.join(sample(list(ascii_uppercase), 5))

    transformation = None
    combiner = None

    try:
        transformation = getattr(LTFArray, 'transform_%s' % transformation_name)
    except AttributeError:
        stderr.write('Transformation %s unknown or currently not implemented\n' % transformation_name)
        quit()

    try:
        combiner = getattr(LTFArray, 'combiner_%s' % combiner_name)
    except AttributeError:
        stderr.write('Combiner %s unknown or currently not implemented\n' % combiner_name)
        quit()

    stderr.write('Learning %s-bit %s XOR Arbiter PUF with %s CRPs and %s restarts.\n\n' % (n, k, N, restarts))
    stderr.write('Using\n')
    stderr.write('  transformation:       %s\n' % transformation)
    stderr.write('  combiner:             %s\n' % combiner)
    stderr.write('  instance random seed: 0x%x\n' % seed_instance)
    stderr.write('  model random seed:    0x%x\n' % seed_model)
    stderr.write('  log file prefix:      %s' % log_file_prefix)
    stderr.write('\n\n')

    experiments = []

    for j in range(instances):

        experiments.append(
            ExperimentLogisticRegression(
                log_name='%s_instance0x%x_model0x%x.log' % (log_file_prefix, seed_instance, seed_model),
                n=n,
                k=k,
                crp_count=N,
                seed_model=seed_model,
                seed_instance=seed_instance,
                transformation=transformation,
                combiner=combiner,
                restarts=restarts,
            )
        )

        seed_instance += 1

    Experimenter(log_file_prefix + '.log', experiments).run()

if __name__ == '__main__':
    main(argv)
