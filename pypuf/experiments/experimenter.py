"""
This module provides a class which acts as process pool. The is build in order to distribute several experiments over
the system cores. The results of the experiments are written into on a single log file which is coordinated by a logging
process. In order to run experiments a path to a logfile and list of experiments is needed which inherit the
pypuf.experiments.experiment.base class.

Example usage:

experiments = []
n = 128
for i in range(n):
    log_name = 'test_multiprocessing_logs{0}'.format(i)
    experiment = Experiment(...)
    experiments.append(experiment)

experimenter = Experimenter('experimenter_log_path', experiments)
experimenter.run()
"""
import multiprocessing
import logging
import pickle
import queue
import sys
import traceback
import os
from datetime import datetime, timedelta


class Experimenter(object):
    """
    Coordinated, parallel execution of Experiments with logging.
    """

    def __init__(self, log_name, experiments, cpu_limit=2**16, auto_multiprocessing=False, update_callback=None):
        """
        :param experiments: A list of pypuf.experiments.experiment.base.Experiment
        :param log_name: A unique file path where to output should be logged.
        :param cpu_limit: Maximum number of parallel processes that run experiments.
        """

        # Store experiments list
        self.experiments = experiments

        # Store logger name
        self.logger_name = log_name

        # Setup parallel execution limit
        self.cpu_limit = min(cpu_limit, multiprocessing.cpu_count())
        self.semaphore = None

        # experimental results
        self.results = []

        # Disable automatic multiprocessing
        if not auto_multiprocessing:
            self.disable_auto_multiprocessing()

        # Callback for new results
        self.update_callback = update_callback

    def run(self):
        """
        Runs all experiments.
        """
        self.semaphore = multiprocessing.BoundedSemaphore(min(self.cpu_limit, len(self.experiments)))

        # Setup multiprocessing logging
        logging_queue = multiprocessing.Queue(-1)
        listener = multiprocessing.Process(target=log_listener,
                                           args=(logging_queue, setup_logger, self.logger_name,))
        listener.start()

        # list of active jobs
        active_jobs = []

        # list of queues
        result_queues = [None] * len(self.experiments)
        self.results = [None] * len(self.experiments)

        start_time = datetime.now()

        def output_status(number_of_started_jobs):
            progress = (number_of_started_jobs - len(active_jobs)) / len(self.experiments)
            elapsed_time = datetime.now() - start_time
            sys.stdout.write(
                "%s %i jobs total, %i finished, %i running, %i queued, progress %.2f, remaining time: %s\n" %
                (
                    datetime.now().strftime('%c'),
                    len(self.experiments),
                    number_of_started_jobs - len(active_jobs),
                    len(active_jobs),
                    len(self.experiments) - number_of_started_jobs,
                    progress,
                    timedelta(seconds=(elapsed_time * (1 - progress) / progress).total_seconds() // 15 * 15) if progress > 0 else '???',
                )
            )

        def list_active_jobs():
            """
            update list of active jobs
            return: [multiprocessing.Process] list of active jobs
            """
            still_active_jobs = []
            from time import sleep
            sleep(.1)  # Wait for jobs to exit after they release. Ugly but good enough for the status output. 🙈
            for j in active_jobs:
                j.join(0)
                if j.exitcode is None:
                    still_active_jobs.append(j)
            return still_active_jobs

        def update_results():
            updated = False
            for (i, result_queue) in enumerate(result_queues):
                if result_queue and not self.results[i]:
                    try:
                        self.results[i] = pickle.loads(result_queue.get_nowait())
                        updated = True
                        result_queue.close()

                    except queue.Empty:
                        pass
            if updated and self.update_callback:
                self.update_callback()
            return updated

        for (i, exp) in enumerate(self.experiments):

            # define experiment process
            def run_experiment(experiment, logging_queue, semaphore, logger_name, result_queue):
                """
                This method is responsible to start the experiment and release the semaphore which coordinates the
                number of parallel running processes.
                :param experiment: pypuf.experiments.experiment.base.Experiment
                                   A implementation of the base experiment class which should be executed
                :param logging_queue: multiprocessing.queue
                              This is used to coordinate the processes in order to obtain logs.
                :param semaphore: multiprocessing.BoundedSemaphore(self.cpu_limit)
                                  A semaphore to limit the number of concurrent running experiments
                :param logger_name: String
                                    Path to or name to the log file of the experiment
                :param result_queue: multiprocessing.queue
                                Used to obtain the result object.
                """
                ex = None
                try:
                    experiment.execute(logging_queue, logger_name, result_queue)  # run the actual experiment
                except Exception as ex_raised:  # pylint: disable=W
                    ex = ex_raised
                semaphore.release()  # release CPU
                result_queue.close()
                if ex:
                    raise ex

            result_queues[i] = multiprocessing.Queue()

            job = multiprocessing.Process(
                target=run_experiment,
                args=(exp, logging_queue, self.semaphore, self.logger_name, result_queues[i])
            )

            # wait for a free CPU
            self.semaphore.acquire()

            # start experiment
            job.start()
            active_jobs = list_active_jobs()
            active_jobs.append(job)

            # output status and update results
            output_status(i + 1)
            update_results()

        # wait for all processes to be finished
        still_running = min(len(active_jobs), self.cpu_limit, len(self.experiments))
        while still_running > 0:
            self.semaphore.acquire()
            still_running -= 1
            active_jobs = list_active_jobs()
            output_status(len(self.experiments))
            update_results()

        # join remaining processes, at this point this will be without wait
        for job in active_jobs:
            job.join()

        # Quit logging process
        logging_queue.put_nowait(None)
        listener.join()

    @staticmethod
    def disable_auto_multiprocessing():
        """
        Disables numpy's automatic multiprocessing/multithreading for the current
        python instance by setting environment variables.
        This method must be called before numpy is first imported. If numpy was
        already imported and the environment was not yet set accordingly, an
        Exception will be raised.
        """
        desired_environment = {
            'OMP_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
        }
        if 'numpy' in sys.modules:
            for key, val in desired_environment.items():
                if key not in os.environ or os.environ[key] != val:
                    raise Exception('Cannot disable numpy\'s automatic parallel computing, '
                                    'because it was already imported. To fix this issue, '
                                    'import Experimenter before numpy or run python in the '
                                    'following environment: ' + str(desired_environment))
        for key, val in desired_environment.items():
            os.environ[key] = val


def setup_logger(logger_name):
    """
    This method is used to open the file handler for a logger_name. The resulting log file will have the format
    'logger_namer.log'.
    :param logger_name: string
                        Path to or name to the log file
    """
    root = logging.getLogger(logger_name)

    # Setup logging to both file and console
    file_handler = logging.FileHandler(filename='logs/%s.log' % logger_name, mode='w')
    file_handler.setLevel(logging.INFO)

    root.addHandler(file_handler)


def log_listener(queue, configurer, logger_name):
    """
    This is the root function of logging process which is responsible to log the experiment results.
    :param queue: multiprocessing.queue
                  This is used to coordinate the processes in order to obtain results.
    :param configurer: A function which setup the logger which logs the messages read from the queue.
    :param logger_name: String
                        Path to or name to the log file
    """
    configurer(logger_name)
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:  # pylint: disable=W
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
