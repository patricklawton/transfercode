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

def main(args):
    """For the particles synthesized by Zhe at the Secanna group"""
    project = signac.init_project('SecannaAssembly')

    # packing_fractions = np.linspace(0.50, 0.65, 4)
    g_vals = np.array([1,5,9.8,12]) #np.arange(1,12,1.5)
    truncations = np.array([0.55]) #np.arange(0.55, 0.76, 0.02)
    run_nums=list(range(1))

    sps = cartesian(g_val=g_vals, truncation=truncations, run_num=run_nums)

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
