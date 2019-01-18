from sys import argv, stderr
import numpy as np

from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf import tools


def main(args):
    if len(args) != 3:
        stderr.write('\nUsage:')
        stderr.write('python calc_intra_dist.py n task')
        stderr.write('  n:	    length of NoisyLTFArray\n')
        stderr.write('  task:   task number\n')
        quit(1)

    n = int(args[1])
    task = int(args[2])
    k = 1
    transform = LTFArray.transform_id
    combiner = LTFArray.combiner_xor
    num = 10
    m = 2000
    means = np.zeros(m)

    for j in range(1, m+1):
        distances = np.zeros(num)
        for i in range(num):
            noisiness = j * 0.001
            instance = NoisyLTFArray(
                weight_array=LTFArray.normal_weights(n, k),
                transform=transform,
                combiner=combiner,
                sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(n=n, sigma_weight=1, noisiness=noisiness)
            )
            dist = tools.approx_dist(
                instance1=instance,
                instance2=instance,
                num=100000
            )
            if dist < .5:
                dist = 1 - dist
            distances[i] = dist
        means[j-1] = np.mean(distances)

        name = 'intra-distances_%i_%i.csv' % (n, task)
        f = open(name, 'w')
        for mean in means:
            f.write(str(mean) + '\n')
        f.close()


if __name__ == '__main__':
    main(argv)
