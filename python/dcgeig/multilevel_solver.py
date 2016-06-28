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
import scipy.sparse as SS

import time



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
    n3 = n - n1 - n2

    K11, K22, K33 = tools.get_submatrices(K, ptree)
    M11, M22, M33 = tools.get_submatrices(M, ptree)

    assert K[:,:n1][n1:n1+n2,:].nnz == 0
    assert M[:,:n1][n1:n1+n2,:].nnz == 0


    # Conquer
    d1, X1, stats1 = \
        impl(options, K11, M11, level+1, left_child)
    d2, X2, stats2 = \
        impl(options, K22, M22, level+1, right_child)


    # Combine
    wallclock_time_start = time.time()
    cpu_time_start = time.clock()

    if n3 > 0:
        d3, X3, _, _ = tools.rayleigh_ritz(K33, M33)
    else:
        d3 = NP.full([n3], 1, dtype=K.dtype)
        X3 = NP.full([n3,n3], 1, dtype=K.dtype)

    d = NP.concatenate([d1, d2, d3])
    X = ML.matrix( scipy.linalg.block_diag(X1, X2, X3) )

    # manually free memory
    K11 = None; K22 = None; K33 = None
    M11 = None; M22 = None; M33 = None
    d1 = None; d2 = None; d3 = None
    X1 = None; X2 = None; X3 = None

    ii = NP.argsort(d)
    d = d[ii]
    X = X[:,ii]
    eta, delta = tools.compute_errors(K, M, d, X)
    ii = None


    if do_terminate(d, X, eta, delta):
        t = select(d, delta)
        d, X, eta, delta = tools.apply_selection(t, d, X, eta, delta)

        return d, X, tools.make_stats_tree(**locals())


    if d.size == n:
        d, X, eta, delta = tools.rayleigh_ritz(K, M)

        t = select(d, delta)
        d, X, eta, delta = tools.apply_selection(t, d, X, eta, delta)

        return d, X, tools.make_stats_tree(**locals())


    wallclock_time_sle = 0
    wallclock_time_rr = 0

    for i in xrange(1, options.max_num_iterations+1):
        wallclock_time_sle_start = time.time()
        t = eta > NP.finfo(K.dtype).eps
        tau = max(d)
        X[:,t] = tools.solve_SLE(ptree, K, (K - tau*M) * X[:,t])
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



def execute(options):
    K = options.K
    M = options.M

    assert utils.is_hermitian(K)
    assert utils.is_hermitian(M)

    # manual memory management
    options.K = None; options.M = None

    G = abs(K) + abs(M)
    ptree, perm = sparse_tools.multilevel_nested_dissection(G, options.n_direct)
    G = None

    K = K[:,perm][perm,:]
    M = M[:,perm][perm,:]

    ptree = sparse_tools.add_postorder_id(ptree)
    ptree = tools.compute_schur_complement(K, ptree)

    level = 0
    d, X, _ = impl(options, K, M, level, ptree)

    eta, delta = tools.compute_errors(K, M, d, X)

    return d, eta, delta
