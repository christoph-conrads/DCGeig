#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# benchmark comparing solver and setup time of
# - SuperLU,
# - SuperLU computing a Cholesky factorization,
# - direct substructuring.

import numpy as NP
import numpy.random
import numpy.matlib as ML

import scipy.linalg as LA
import scipy.sparse as SS
import scipy.sparse.linalg as SL

import scipy.io as IO

import dcgeig.direct_substructuring as DS
import dcgeig.metis as metis
import dcgeig.utils as utils
import dcgeig.sparse_tools as sparse_tools
from dcgeig.sparse_tools import Tree

import sys
import os.path as OP

import time



DOUBLE_BYTES = 8
MAX_TOL = 10.0**(-10)


def normalize_path(filename):
    assert isinstance(filename, str)

    path = OP.expanduser(filename)
    abs_path = path if OP.isabs(path) else OP.abspath(path)

    return abs_path



def compute_backward_error(A, X, B):
    n = A.shape[0]
    v0 = NP.ones( [n,1], dtype=A.dtype )
    s = SL.eigsh(A, k=1, v0=v0, ncv=8, return_eigenvectors=False)[0]
    R = A*X - B

    norm = lambda V: LA.norm(V, axis=0)

    eta = norm(R) / ( s * norm(X) + norm(B) )

    return eta



def measure_time(f):
    t0 = time.time()
    c0 = time.clock()
    ret = f()
    c1 = time.clock()
    t1 = time.time()

    return (t1-t0, c1-c0) + ret



def measure_solve_times(filename, solver, A, ms, solve):
    assert isinstance(filename, str)
    assert isinstance(solver, str)
    assert utils.is_hermitian(A)


    n = A.shape[0]

    l = len(ms)
    t_solve = NP.full(l, NP.nan, dtype=A.dtype)
    c_solve = NP.full(l, NP.nan, dtype=A.dtype)

    for i in xrange(l):
        m = ms[i]

        try:
            random = numpy.random.RandomState(seed=1)
            B = ML.matrix( 2 * random.rand(n, m) - 1, dtype=A.dtype )

            t, c, X = measure_time( lambda: (solve(B),) )
            #assert max( compute_backward_error(A, X, B) ) < MAX_TOL

            t_solve[i] = t
            c_solve[i] = c
        except MemoryError:
            fmt = '{:19s} {:2s} n_rhs={:d}: {:s}\n'
            sys.stderr.write( fmt.format(filename, solver, m, 'out of memory') )
            break
        except Exception as e:
            fmt = '{:19s} {:2s} n_rhs={:d}: {:s}\n'
            sys.stderr.write( fmt.format(filename, solver, m, str(e)) )

    return t_solve, c_solve



def compute_nnz_schur(ptree):
    if Tree.is_leaf_node(ptree):
        n = ptree.n
        return n*n

    n = ptree.n
    n1 = ptree.left_child.n
    n2 = ptree.right_child.n
    n3 = n - n1 - n2
    assert n3 >= 0

    mem = \
        n3*n3 + \
        compute_nnz_schur(ptree.left_child) + \
        compute_nnz_schur(ptree.right_child)

    return mem



def make_schur(A, n_direct=1024):
    ptree, perm = sparse_tools.multilevel_nested_dissection(A, n_direct)

    App = A[:,perm][perm,:]
    ptree = DS.setup(ptree, App)

    def solve(B):
        Bp = B[perm,:]
        Xp = DS.solve_SLE(ptree, App, Bp)
        X = 0*B
        X[perm,:] = Xp

        return X


    return ptree, solve



def benchmark_direct_substructuring(filename, A, ms, n_direct):
    assert utils.is_hermitian(A)
    assert isinstance(n_direct, int)
    assert n_direct > 0

    n = A.shape[0]

    def setup():
        return make_schur(A, n_direct)

    t_setup, c_setup, ptree, solve = measure_time(setup)
    t_solve, c_solve = measure_solve_times(filename, 'DS', A, ms, solve)

    nnz = compute_nnz_schur(ptree)

    stat = {
        'solver': 'DS',
        'nnz': nnz,
        't_setup': t_setup,
        'c_setup': c_setup,
        't_solve': t_solve,
        'c_solve': c_solve
    }

    return stat



def benchmark_superLU(filename, A, ms, solver, superLU_options={}):
    assert utils.is_hermitian(A)

    n = A.shape[0]

    def setup():
        LU = SL.splu(A, **superLU_options)
        return (LU,)

    t_setup, c_setup, LU = measure_time(setup)
    t_solve, c_solve = measure_solve_times(filename, solver, A, ms, LU.solve)

    stat = {
        'solver': solver,
        'nnz': LU.nnz,
        't_setup': t_setup,
        'c_setup': c_setup,
        't_solve': t_solve,
        'c_solve': c_solve
    }

    return stat



def main(argv):
    if len(argv) != 2:
        script = argv[0] if len(argv) > 0 else 'a.out'
        print 'usage: python {} <mtx file>'.format(script)
        return 0

    path = normalize_path(argv[1])
    filename = OP.basename(path)

    A = IO.mmread(path)
    A = SS.csc_matrix(A)
    n = A.shape[0]

    n_direct = 1024


    if n <= n_direct:
        return 0


    # balance matrix
    I = SS.identity(n, dtype=A.dtype)
    s, D = sparse_tools.balance_matrix_pencil(A, I)
    A = SS.csc_matrix(D*A*D)


    # get benchmarking data
    solvers = {
        'DS': lambda: \
            benchmark_direct_substructuring(filename, A, ms, n_direct),
        'LU': lambda: \
            benchmark_superLU(filename, A, ms, 'LU'),
        'LL': lambda: \
            benchmark_superLU(filename, A, ms, 'LL', {'diag_pivot_thresh': 0.0})
    }

    ms = 2**NP.arange(8, 12, dtype=int)


    def call_solver(name):
        fmt = '{:19s} {:2s}: {:s}\n'

        try:
            f = solvers[name]
            return f()
        except MemoryError:
            sys.stderr.write( fmt.format(filename, name, 'out of memory') )
        except Exception as e:
            sys.stderr.write( fmt.format(filename, name, str(e)) )

        l = len(ms)
        stat = {
            'solver': name,
            'nnz': -1,
            't_setup': NP.nan,
            'c_setup': NP.nan,
            't_solve': NP.full(l, NP.nan, dtype=A.dtype),
            'c_solve': NP.full(l, NP.nan, dtype=A.dtype)
        }

        return stat


    stats = map(call_solver, solvers)

    header = \
        '{:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s}'
    fmt = \
        '{:19s} {:6d} {:2s} {:4d} {:4.1f} {:6.1f} {:6.1f} {:4d} {:6.1f} {:6.1f}'

    print header.format( \
        'problem', 'n', 'solver',
        'mem(MB)', 'fill-in',
        't-setup', 'c-setup',
        'n_rhs', 't-solve', 'c-solve')

    for stat in stats:
        solver = stat['solver']

        nnz = stat['nnz']
        fill_in = 1.0 * nnz / A.nnz
        mem_MB = DOUBLE_BYTES * nnz / (2**20)

        t_setup = stat['t_setup']
        c_setup = stat['c_setup']

        for i in xrange(len(ms)):
            n_rhs = ms[i]
            t_solve = stat['t_solve'][i]
            c_solve = stat['c_solve'][i]

            text = fmt.format( \
                filename, n, solver,
                mem_MB, fill_in,
                t_setup, c_setup,
                n_rhs, t_solve, c_solve)

            print text


    return 0



if __name__ == '__main__':
    sys.exit( main(sys.argv) )
