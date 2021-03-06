"""
Helper module to run studies as defined in pypuf.studies
"""
import sys
import argparse

from pypuf.experiments import Experimenter
from pypuf.tools import find_study_class


def main(args):
    """
    Tries to find the study specified and runs it.
    """
    parser = argparse.ArgumentParser(
        prog='study',
        description="Runs a pypuf study",
    )
    parser.add_argument("study", help="name of the study to be run", type=str)

    args = parser.parse_args(args)

    study_class = find_study_class(args.study)
    print('Running {}.{}'.format(study_class.__module__, study_class.__name__))
    study = study_class()
    study.run()


if __name__ == '__main__':
    Experimenter.disable_auto_multiprocessing()
    main(sys.argv[1:])
