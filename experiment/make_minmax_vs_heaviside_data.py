#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as NP
import numpy.polynomial.chebyshev as NPC

import numbers

import sys



def compute_jackson_coefficients(d):
    assert isinstance(d, int)
    assert d >= 0

    if d == 0:
        return NP.full(1, 1.0)

    k = 1.0 * NP.arange(0, d+1)
    r = NP.pi / (d+1)
    cs = (d+1-k) * NP.cos(k*r) + NP.sin(k*r) / NP.tan(r)
    cs[-1] = 0

    assert cs.size == d+1

    return cs / (d+1)



def compute_chebyshev_heaviside_coefficients(d):
    assert isinstance(d, int)
    assert d >= 0

    k = 1.0 * NP.arange(1, d+1)
    cs = NP.full(d+1, NP.nan)
    cs[0] = 0.5
    cs[1::4] = 1.0
    cs[2::4] = 0.0
    cs[3::4] = -1.0
    cs[4::4] = 0.0
    cs[1:] = cs[1:] * 2/(NP.pi * k)

    return cs



def main(argv):
    # parse arguments
    aout = argv[0]

    if len(argv) != 5:
        fmt = 'usage: python {:s} <degree> <norm> <l> <r>\n'
        sys.stderr.write( fmt.format(aout) )
        return 1

    for i in range(2, len(argv)):
        fmt = '{:s}: argument {:d} must be finite, positive (value=%f)'
        f = float(argv[i])

        if NP.isinf(f) or NP.isnan(f) or f <= 0:
            sys.stderr.write( fmt.format(aout, i, f) )
            return 2

    d = int(argv[1])

    if d <= 0:
        return 3

    x_normal = float(argv[2])
    l = float(argv[3])
    r = float(argv[4])

    if l >= r:
        return 4


    # set up polynomial
    a = 0.0
    b = 1.0/(2*x_normal)
    c = (b + a) / 2
    e = (b - a) / 2
    assert c > 0
    assert e > 0

    xs = NP.linspace(l, r, 1000)
    cs = compute_chebyshev_heaviside_coefficients(d)
    js = compute_jackson_coefficients(d)
    ps = NPC.chebtrim(cs * js)

    qs = NP.append( NP.full(len(ps)-1, 0.0), NP.full(1, 1.0) )

    eval_poly = lambda xs, ps: NPC.chebval( (1/xs-c)/e, ps )
    f = lambda xs: eval_poly(xs, ps) / eval_poly(x_normal, ps)
    g = lambda xs: eval_poly(xs, qs) / eval_poly(x_normal, qs)

    ys = abs( f(xs) )
    zs = abs( g(xs) )


    # print data
    comment = '# degree={:d} normalize_at={:8.2e}'
    header = '{:>10s}  {:>10s} {:>10s}'
    fmt = '{:10.4e}  {:10.4e} {:10.4e}'

    print comment.format(len(qs)-1, x_normal)
    print header.format('x', 'min-max', 'heaviside')

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        z = zs[i]

        print fmt.format(x, z, y)



if __name__ == '__main__':
    sys.exit( main(sys.argv) )
