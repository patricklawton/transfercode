#!/usr/bin/env python
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""
import logging
import argparse
import itertools

import signac
import numpy as np

def cartesian(**kwargs):
    for combo in itertools.product(*kwargs.values()):
        yield dict(zip(kwargs.keys(), combo))

def round_wrap(arr, precision=1e-3):
    return [round(float(x), -int(np.log10(precision))) for x in arr]
def main(args):
    """For the particles synthesized by Zhe at the Secanna group"""
    project = signac.init_project('SecannaAssembly')

    pfs = round_wrap(np.linspace(0.55, 0.60, 5, endpoint=True))
    truncations = round_wrap(np.arange(0.47, 0.84, 0.02))
    structures = ['cubicdiamond', 'hexagonaldiamond']

    sps = cartesian(truncation=truncations, pf=pfs, structure=structures)

    for sp in sps:
        project.open_job(sp).init()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Initialize the data space.")
    parser.add_argument(
        '-n', '--num-replicas',
        type=int,
        default=1,
        help="Initialize multiple replications.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args)
