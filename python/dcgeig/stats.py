#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.


from dcgeig.sparse_tools import Tree

import numpy as NP

import copy


def get_header():
    fmt = ( \
        '{:>3s} {:>4s} {:3s} '
        '{:>6s} {:>4s} {:>4s} {:>4s}  '
        '{:>8s} {:>8s} {:>8s} {:>8s}  '
        '{:>8s} {:>7s} {:>8s}  '
        '{:>8s} {:>7s} {:>8s}  '
        '{:>8s} {:>7s} {:>8s} '
        '{:>2s} {:>5s} {:>6s} {:>6s} {:>6s}\n')

    header = fmt.format( \
        'pid', 'sid', 'lvl',
        'n', 'n_c', 'n_s', 'fill',
        'norm:K', 'norm:M', 'norm:K12', 'norm:M12',
        'min:ev', 'max:ev', 'median:ev',
        'min:be', 'max:be', 'median:be',
        'min:fe', 'max:fe', 'median:fe',
        'iter', 't-sle', 't-rr', 't-wc', 't-cpu')

    return header



def add_id(node, id0=1):
    assert isinstance(id0, int)
    assert id0 > 0

    if Tree.is_leaf_node(node):
        new_node = copy.copy(node)
        new_node.id = id0
        return new_node, id0+1

    left, id1 = add_id(node.left_child, id0)
    right,id2 = add_id(node.right_child, id1)

    new_node = copy.copy(node)
    new_node.id = id2
    new_node.left_child = left
    new_node.right_child = right

    return new_node, id2+1



def to_string(stats):
    assert isinstance(stats, list)

    line_fmt = (\
        '%3d %4d %3d ' # subproblem id level
        '%6d %4d %4d %4.1f  ' # n n_c n_s fill(LU)
        '%8.2e %8.2e %8.2e %8.2e  ' # norm(K) norm(M) norm(K12) norm(M12)
        '%8.2e %8.2e %8.2e  ' # eigenvalue statistics
        '%8.2e %8.2e %8.2e  ' # backward error statistics
        '%8.2e %8.2e %8.2e  ' # relative forward error statistics
        '%2d %6.1f %6.1f %6.1f %6.1f\n') # num_iterations, timing information


    stats = map(lambda s: add_id(s)[0], stats)


    def get_statistics(xs):
        return NP.min(xs), NP.max(xs), NP.median(xs)

    def f(pid, level, x):
        assert isinstance(x, Tree)
        assert level == x.level

        d_rel_stats = get_statistics(x.d_rel)
        eta_stats = get_statistics(x.eta)
        rfe_stats = get_statistics(x.rfe)

        assert x.nnz_LU != 0
        fill_in = -1 if x.nnz_LU < 0 else 1.0 * x.nnz_LU / x.nnz_K

        line = line_fmt % (
            pid, x.id, level,
            x.n, x.n_c, x.n_s, fill_in,
            x.norm_K, x.norm_M, x.norm_K12, x.norm_M12,
            d_rel_stats[0], d_rel_stats[1], d_rel_stats[2],
            eta_stats[0], eta_stats[1], eta_stats[2],
            rfe_stats[0], rfe_stats[1], rfe_stats[2],
            x.num_iterations,
            x.wallclock_time_sle, x.wallclock_time_rr,
            x.wallclock_time, x.cpu_time)

        if Tree.is_leaf_node(x):
            return line

        l = f(pid, level+1, x.left_child)
        r = f(pid, level+1, x.right_child)

        return l+r+line


    strings = map( lambda i: f(i, 0, stats[i]), xrange(len(stats)) )

    return ''.join(strings)
