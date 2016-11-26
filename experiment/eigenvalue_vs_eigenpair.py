#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numpy as NP

import scipy.linalg as SL
import scipy.sparse as SS

import scipy.io as IO

import sys
import os.path as OP

import time



def normalize_path(filename):
    assert isinstance(filename, str)

    path = OP.expanduser(filename)
    abs_path = path if OP.isabs(path) else OP.abspath(path)

    return abs_path



def main(argv):
    if len(argv) != 2:
        script = argv[0] if len(argv) > 0 else 'a.out'
        print 'usage: python {} <mtx file>'.format(script)
        return 0

    k_path = normalize_path(argv[1])
    k_dirname = OP.dirname(k_path)
    k_basename = OP.basename(k_path)
    assert k_basename != ''

    K = IO.mmread(k_path)
    M = SS.identity(K.shape[0])

    if str.count(k_basename, 'k') == 1:
        m_basename = str.replace(k_basename, 'k', 'm')
        m_path = OP.join(k_dirname, m_basename)

        try:
            M = IO.mmread(m_path).astype(K.dtype)
        except IOError as e:
            if e.errno != 2:
                raise

    K = K.todense()
    M = M.todense()
    n = K.shape[0]


    # this takes too long
    if n > 10000:
        return 0


    # dummy call, test if M is singular
    try:
        d = SL.eigvalsh(K, M)
    except:
        return 0


    # measure
    c0 = time.clock()
    t0 = time.time()
    d_max = SL.eigvalsh(K, M, eigvals=(n-1,n-1))
    c1 = time.clock()
    t1 = time.time()
    d = SL.eigvalsh(K, M)
    c2 = time.clock()
    t2 = time.time()
    d, X = SL.eigh(K, M)
    c3 = time.clock()
    t3 = time.time()


    c_dmax = c1-c0
    t_dmax = t1-t0
    c_d = c2-c1
    t_d = t2-t1
    c_all = c3-c2
    t_all = t3-t2


    # print results
    fmt = '{:20s} {:4d}  {:6.1f} {:6.1f}  {:6.1f} {:6.1f}  {:6.1f} {:6.1f}'
    print fmt.format(k_basename, n, t_dmax, c_dmax, t_d, c_d, t_all, c_all)

    return 0



if __name__ == '__main__':
    sys.exit( main(sys.argv) )
