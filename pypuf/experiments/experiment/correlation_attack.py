from numpy.random import RandomState
from numpy.linalg import norm
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.regression.correlation_attack import CorrelationAttack
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import TrainingSet, approx_dist


class ExperimentCorrelationAttack(Experiment):

    def __init__(self, n, k,
                 log_name,
                 seed_model,
                 seed_instance,
                 seed_challenge,
                 seed_challenge_distance,
                 N,
                 ):
        super().__init__(
            log_name='%s.0x%x_0x%x_0_%i_%i_%i_%s_%s' % (
                log_name,
                seed_model,
                seed_instance,
                n,
                k,
                N,
                LTFArray.transform_lightweight_secure_original.__name__,
                LTFArray.combiner_xor.__name__,
            ),
        )
        self.n = n
        self.k = k
        self.N = N
        self.seed_instance = seed_instance
        self.instance_prng = RandomState(seed=self.seed_instance)
        self.seed_model = seed_model
        self.model_prng = RandomState(seed=self.seed_model)
        self.combiner = LTFArray.combiner_xor
        self.transformation = LTFArray.transform_lightweight_secure_original
        self.seed_challenge = seed_challenge
        self.challenge_prng = RandomState(self.seed_challenge)
        self.seed_chl_distance = seed_challenge_distance
        self.distance_prng = RandomState(self.seed_chl_distance)
        self.instance = None
        self.learner = None
        self.model = None

    def run(self):
        # TODO input transformation is computed twice. Add a shortcut to recycle results from the first computation
        self.instance = LTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, random_instance=self.instance_prng),
            transform=self.transformation,
            combiner=self.combiner,
        )
        self.learner = CorrelationAttack(
            n=self.n,
            k=self.k,
            training_set=TrainingSet(instance=self.instance, N=self.N, random_instance=self.challenge_prng),
            validation_set=TrainingSet(instance=self.instance, N=2000, random_instance=self.distance_prng),
            weights_prng=self.model_prng,
            logger=self.progress_logger,
        )
        self.model = self.learner.learn()

    def analyze(self):
        """
        Analyzes the learned result.
        """
        assert self.model is not None

        self.result_logger.info(
            # seed_instance  seed_model n      k      N      time   initial_iterations initial_accuracy best_accuracy
            '0x%x\t'        '0x%x\t'   '%i\t' '%i\t' '%i\t' '%f\t' '%i\t'             '%f\t'           '%f\t'
                # accuracy model values  best_iteration  rounds
                '%f\t'    '%s\t'        '%i'              '%i',
            self.seed_instance,
            self.seed_model,
            self.n,
            self.k,
            self.N,
            self.measured_time,
            self.learner.initial_iterations,
            self.learner.initial_accuracy,
            self.learner.best_accuracy,
            1.0 - approx_dist(
                self.instance,
                self.model,
                min(10000, 2 ** self.n),
                random_instance=self.distance_prng,
            ),
            ','.join(map(str, self.model.weight_array.flatten() / norm(self.model.weight_array.flatten()))),
            self.learner.best_iteration,
            self.learner.rounds,
        )
