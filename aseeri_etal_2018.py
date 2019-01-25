from pypuf.experiments.experimenter import Experimenter
from pypuf.experiments.experiment.mlp import ExperimentMLP
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


experiments = []
log = 'Aseeri_etal_2018'
e = Experimenter(log, experiments)

configurations = [
    ( 64, 4,   .4),
    ( 64, 5,   .8),
    ( 64, 6,  2),
    ( 64, 7,  5),
    ( 64, 8, 30),
    (128, 4,  .8),
    (128, 5,  3),
    (128, 6, 20),
    (128, 7, 40),
]

for (n, k, N) in configurations:
    for i in range(10):
        experiments.append(
            ExperimentMLP(
                log_name=log,
                n=n,
                k=k,
                N=int(N * 10e6),
                seed_simulation=0x1 + i,
                seed_model=0x1000 + i,
                transformation=LTFArray.transform_id,
                combiner=LTFArray.combiner_xor,
                seed_challenge=0x2 + i,
                seed_accuracy=0x3 + i,
                batch_size=1000 if k < 6 else 10000,
            )
        )

e.run()
