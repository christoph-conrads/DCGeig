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
    a = 5.0 * lambda_c;
    b = 10.0 * lambda_c;

    tau = 1/lambda_c
    c, e = get_ellipse(1/b, 1/a)

    xs = NP.linspace(lambda_1, 2*b, num=5000 )
    assert NP.all( xs > 0 )

    for k in range(1, 4):
        ys = poly(k, c, e, tau, 1/xs) / poly(k, c, e, tau, 1/lambda_c)

        fmt='cheb({:d}, {:.1f}, {:.1f})'
        PP.plot(xs, abs(ys), label=fmt.format(k, a, b))


    cs = cayley(xs, 10*lambda_c) / cayley(lambda_c, 10*lambda_c)
    PP.plot(xs, abs(cs), label='cayley')

    PP.plot(xs, lambda_c/xs, label='inv')

    PP.plot( (lambda_c,lambda_c), (0, 1e5), 'k-' )
    PP.plot( (10*lambda_c,10*lambda_c), (0, 1e5), 'k-' )

    t = xs >= lambda_c

    axes = PP.gca()
    axes.set_xlim([lambda_1, 12*lambda_c])
    PP.yscale('log')

    PP.legend()
    PP.show()



if __name__ == '__main__':
    sys.exit(main(sys.argv))
