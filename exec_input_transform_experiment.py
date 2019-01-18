from numpy.random import RandomState

from pypuf.experiments.experiment.input_transform_experiment import InputTransformExperiment
from pypuf.experiments.experimenter import Experimenter
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


experiments = list()

k = 1
n = 128
log_name = 'trafos_k%i_n%i' % (k, n)
params = list()
for i in range(200, 4200, 200):
    for j in range(100):
        params.append([k, n, LTFArray.transform_id, LTFArray.combiner_xor, i, None, None])
        params.append([k, n, LTFArray.transform_atf, LTFArray.combiner_xor, i, None, None])
        params.append([k, n, LTFArray.transform_aes_sbox, LTFArray.combiner_xor, i, None, None])
        params.append([k, n, LTFArray.transform_lightweight_secure_original, LTFArray.combiner_xor, i, None, None])
        if n == 64:
            params.append([k, n, LTFArray.transform_fixed_permutation, LTFArray.combiner_xor, i, None, None])
        params.append([k, n, LTFArray.transform_random, LTFArray.combiner_xor, i, None, None])

for i in range(len(params)):
    experiment = InputTransformExperiment(
        log_name=log_name + '_%i' % i,
        k=params[i][0],
        n=params[i][1],
        transform=params[i][2],
        combiner=params[i][3],
        num=params[i][4],
        seed_instance=RandomState(params[i][5]).randint(2**32),
        seed_challenges=RandomState(params[i][6]).randint(2**32)
    )
    experiments.append(experiment)

experimenter = Experimenter(experiments=experiments, log_name=log_name)
experimenter.run()
