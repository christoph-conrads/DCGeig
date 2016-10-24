#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import sys

import numpy as NP
import numpy.polynomial.polynomial as NPP
import numpy.polynomial.chebyshev as NPC

import numbers



def compute_jackson_coefficients(d):
    assert isinstance(d, int)
    assert d >= 0

    if d == 0:
        return NP.full_like(xs, 1)

    k = 1.0 * NP.arange(0, d+1)
    r = NP.pi / (d+1)
    cs = (1 - k/(d+1)) * NP.cos(k*r) + NP.sin(k*r) / ((d+1) * NP.tan(r))

    assert cs.size == d+1

    return cs




def make_step_function(a, b):
    assert isinstance(a, numbers.Real)
    assert isinstance(b, numbers.Real)

    assert a < b

    def f(xs):
        if not isinstance(xs, NP.ndarray):
            return 1 if a <= xs <= b else 0

        t = (xs >= a) & (xs <= b)
        ys = NP.empty_like(xs)
        ys[t] = 1
        ys[~t] = 0

        return ys

    return f



def main(argv):
    h = make_step_function(-0.5, +0.5)
    n = 1000
    xs = NP.linspace(-1, +1, n)
    ys = h(xs)

    d = 50
    pc = NPC.chebfit(xs, ys, d)
    pj = pc * compute_jackson_coefficients(d)

    fc = lambda xs: NPC.chebval(xs, pc)
    fj = lambda xs: NPC.chebval(xs, pj)

    yc = fc(xs)
    yj = fj(xs)


    header = '{:24s} {:24s} {:24s}'
    fmt = '{:+.17e} {:+.17e} {:+.17e}'

    print header.format('x', 'chebyshev', 'chebyshev-jackson')
    for i in range(0, n):
        print fmt.format( xs[i], yc[i], yj[i] )



if __name__ == '__main__':
    sys.exit(main(sys.argv))
