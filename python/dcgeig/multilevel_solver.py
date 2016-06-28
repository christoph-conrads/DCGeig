#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import dcgeig.utils as utils
import dcgeig.multilevel_tools as tools
import dcgeig.sparse_tools as sparse_tools
from dcgeig.sparse_tools import Tree

import numpy as NP
import numpy.matlib as ML

import scipy.linalg
import scipy.sparse.linalg as LA

import time



def get_submatrices(A, n1, n2):
    assert utils.is_hermitian(A)
    assert isinstance(n1, int)
    assert isinstance(n2, int)

    # submatrices are never empty
    assert n1 > 0
    assert n2 > 0
    assert A.shape[0] == n1 + n2

    A11 = A[:,:n1][:n1,:]
    A22 = A[:,n1:][n1:,:]

    return A11, A22



def impl(options, K, M, level, ptree):
    n = K.shape[0]

    assert isinstance(level, int)
    assert level >= 0
    assert ptree.n == n

    select = tools.make_eigenpair_selector(options, level)
    do_terminate = tools.make_termination_test(options, level)


    # solve directly?
    if n <= options.n_direct:
        assert Tree.is_leaf_node(ptree)

        wallclock_time_start = time.time()
        cpu_time_start = time.clock()

        d, X, eta, delta = tools.rayleigh_ritz(K, M)

        t = select(d, delta)
        d, X, eta, delta = tools.apply_selection(t, d, X, eta, delta)

        return d, X, tools.make_stats_tree(**locals())


    # use divide-and-conquer
    assert not Tree.is_leaf_node(ptree)
    assert Tree.has_left_child(ptree)
    assert Tree.has_right_child(ptree)

    # Divide
    left_child = ptree.left_child
    right_child = ptree.right_child
    n1 = left_child.n
    n2 = right_child.n

    K11, K22 = get_submatrices(K, n1, n2)
    M11, M22 = get_submatrices(M, n1, n2)


    # Conquer
    d1, X1, stats1 = \
        impl(options, K11, M11, level+1, left_child)
    d2, X2, stats2 = \
        impl(options, K22, M22, level+1, right_child)


    # Combine
    wallclock_time_start = time.time()
    cpu_time_start = time.clock()

    d = NP.concatenate([d1, d2])
    X = ML.matrix( scipy.linalg.block_diag(X1, X2) )

    # manually free memory
    d1 = None; d2 = None
    X1 = None; X2 = None

    ii = NP.argsort(d)
    d = d[ii]
    X = X[:,ii]
    eta, delta = tools.compute_errors(K, M, d, X)
    del ii


    if do_terminate(d, X, eta, delta):
        t = select(d, delta)
        d, X, eta, delta = tools.apply_selection(t, d, X, eta, delta)

        return d, X, tools.make_stats_tree(**locals())


    if d.size == n:
        d, X, eta, delta = tools.rayleigh_ritz(K, M)

        t = select(d, delta)
        d, X, eta, delta = tools.apply_selection(t, d, X, eta, delta)

        return d, X, tools.make_stats_tree(**locals())


    LU = LA.splu(K)

    wallclock_time_sle = 0
    wallclock_time_rr = 0

    for i in xrange(1, options.max_num_iterations+1):
        wallclock_time_sle_start = time.time()
        t = eta > NP.finfo(K.dtype).eps
        tau = max(d)
        X[:,t] = LU.solve((K - tau*M) * X[:,t])
        wallclock_time_sle += time.time() - wallclock_time_sle_start

        wallclock_time_rr_start = time.time()
        d, X, eta, delta = tools.rayleigh_ritz(K, M, X)
        wallclock_time_rr += time.time() - wallclock_time_rr_start

        if do_terminate(d, X, eta, delta):
            break

    assert not NP.any(NP.isinf(d))
    assert not NP.any(NP.isnan(d))

    t = select(d, delta)
    d, X, eta, delta = tools.apply_selection(t, d, X, eta, delta)

    return d, X, tools.make_stats_tree(num_iterations=i, **locals())



def get_ordering(options, K, M):
    G = sparse_tools.matrix_pencil_to_graph(K, M, options.w)
    dim_tree, perm = sparse_tools.multilevel_bisection(G,options.n_direct)

    ptree = sparse_tools.add_postorder_id(dim_tree)

    return ptree, perm



def execute(options, K, M, ptree):
    level = 0

    return impl(options, K, M, level, ptree)
