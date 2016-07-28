#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import sys

import numpy as NP
import matplotlib.pyplot as PP

import numbers


def C(k, t):
    assert isinstance(k, int)
    assert k >= 0

    if k == 0:
        return 1
    if k == 1:
        return t

    return 2 * t * C(k-1, t) - C(k-2, t)



def poly(k, c, e, tau, xs):
    assert isinstance(k, int)
    assert k >= 0
    assert isinstance(c, numbers.Real)
    assert isinstance(e, numbers.Real)
    assert e > 0
    assert isinstance(tau, numbers.Real)
    assert abs(tau - c) > e

    return C(k, (xs-c)/e) / C(k, (tau-c)/e)



def get_ellipse(a, b):
    assert isinstance(a, numbers.Real)
    assert isinstance(b, numbers.Real)
    assert a < b

    c = (a + b) / 2.0
    e = (b - a) / 2.0

    return c, e



def cayley(xs, p):
    return 1/xs * (xs - p)



# poly(k, c, e, tau, t) evaluated iteratively
def chebychev(degree, c, e, tau, t):
    assert isinstance(degree, int)
    assert degree >= 0
    assert isinstance(c, numbers.Real)
    assert c > 0
    assert isinstance(e, numbers.Real)
    assert e > 0
    assert isinstance(tau, numbers.Real)
    assert abs(tau - c) > e

    a = NP.full(degree, NP.nan)
    a[0] = e / (tau - c)

    for k in xrange(degree-1):
        a[k+1] = 1 / (2/a[0] - a[k])

    p0 = NP.full_like(t, 1)
    p1 = (t - c) / (tau - c)

    for k in xrange(degree-1):
        p2 = 2*a[k+1]/e * (t-c) * p1 - a[k+1]*a[k] * p0

        p0 = p1
        p1 = p2
        del p2

    return p1



def main(argv):
    lambda_1 = 0.05;
    lambda_c = 1.0;
    a = 10.0 * lambda_c;
    b = NP.sqrt(2) * 10.0 * lambda_c;

    tau = 1/lambda_c
    c, e = get_ellipse(1/b, 1/a)

    left = lambda_1
    right = 30*lambda_c

    x0 = a
    xs = NP.linspace(left, right, num=1000)
    assert NP.all( xs > 0 )

    chebychev_label='chebychev({:d}, {:.1f}, {:.1f})'

    for k in range(1, 4):
        ys = poly(k, c, e, tau, 1/xs) / poly(k, c, e, tau, 1/x0)

        PP.plot(xs, abs(ys), label=chebychev_label.format(k, a, b))


    cs = 50 * cayley(xs, a) / cayley(lambda_c, a)
    PP.plot(xs, abs(cs), label='cayley')

    PP.plot(xs, x0/xs, label='1/x')

    inf = float('inf')
    c_inf = (1/a + 1/inf) / 2
    e_inf = (1/a - 1/inf) / 2
    k = 2
    zs = C(k, (1/xs - c_inf) / e_inf) / C(k, (1/x0 - c_inf) / e_inf)
    PP.plot( xs, abs(zs), label=chebychev_label.format(k, a, inf) )

    PP.plot( (lambda_c,lambda_c), (0, 1e11), 'k-' )
    PP.plot( (a, a), (0, 1e11), 'k-' )

    axes = PP.gca()
    axes.set_xlim([left, right])
    axes.set_ylim([1e-3, 1e6])
    PP.yscale('log')

    PP.legend()
    PP.grid()
    PP.show()



if __name__ == '__main__':
    sys.exit(main(sys.argv))
