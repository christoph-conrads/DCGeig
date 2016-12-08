#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2015 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as N
import numpy.matlib as M
import numpy.linalg as NL
import numpy.random as R

import scipy.linalg as L
import scipy.optimize as opt

import utils as U
import error_analysis as EA

import gep_solvers

import scipy.io as IO

import sys



def vec2tril(x):
    assert len(x.shape) == 1

    m = len(x)
    n = int( N.sqrt(2*m + 0.25) - 0.5 )

    L = M.matrix( N.full([n,n], 0, dtype=x.dtype) )
    L[ N.tril_indices(n) ] = x;

    assert type(L) == M.matrix

    return L



def fn(solver, x):
    assert len(x.shape) == 1

    m = len(x)
    assert m%2 == 0

    L1 = vec2tril( x[:m/2] )
    L2 = vec2tril( x[m/2:] )

    A = U.force_hermiticity(L1 * L1.H)
    B = U.force_hermiticity(L2 * L2.H)

    theta, X = solver(A, B)

    return max(EA.compute_backward_error(A, B, X, theta))



def main(argc, argv):
    solvers = {
        'standard': gep_solvers.standard,
        'gsvd': gep_solvers.gsvd,
        'qr_csd': gep_solvers.qr_csd,
        'deflation': gep_solvers.deflation
    }

    if argc < 2:
        print 'usage: python {0} <solver> [n=4]'.format(argv[0])
        return 1

    solver_name = argv[1]
    n = 4 if argc < 3 else int(argv[2])

    if solver_name not in solvers:
        print '{0}: unknown solver {1}'.format(argv[0], solver_name)
        return 2

    if n < 1:
        print '{0}: n must be positive'.format(argv[0])
        return 3


    solver = solvers[solver_name]

    m = n * (n+1) / 2

    dtype = N.float64
    eps = N.finfo(dtype).eps
    inf = dtype(N.inf)

    r = R.RandomState(seed=1)
    solver_options = {'maxiter': 1000 + 250*n}

    x_max = None
    z = inf

    for i in range(256 * max(m/36, 1)):
        x0 = r.uniform(low=-1, high=+1, size=2*m)

        ret = opt.minimize( \
            lambda x: 1.0/fn(solver, x), x0, \
            method='Nelder-Mead', \
            options=solver_options)

        if ret.fun < z:
            z = ret.fun
            x_max = ret.x

            y = 1.0 / z / (n * eps)
            print 'i={0:3} j={1:4} z={2:.3e}'.format(i, ret.nit, y)

    L1 = vec2tril(x_max[:m])
    L2 = vec2tril(x_max[m:])

    filename = 'nm_tril_{0}_n={1}_'.format(solver_name, n)
    comment ='Nelder-Mead {0} n={1}'.format(solver_name, n)

    IO.mmwrite(filename+'stiff.mtx', L1, comment=comment)
    IO.mmwrite(filename+'mass.mtx', L2, comment=comment)

    return 0



if __name__=='__main__':
    sys.exit( main(len(sys.argv), sys.argv) )
