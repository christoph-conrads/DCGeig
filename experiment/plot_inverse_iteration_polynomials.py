#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import sys

import numpy as NP
import numpy.polynomial.chebyshev as NPC

import matplotlib.pyplot as PP

import numbers


def get_ellipse(a, b):
    assert isinstance(a, numbers.Real)
    assert isinstance(b, numbers.Real)
    assert a < b

    c = (a + b) / 2.0
    e = (b - a) / 2.0

    return c, e



def make_chebyshev_polynomial(k, a, b):
    assert isinstance(k, int)
    assert k >= 0

    coeffs = NP.zeros(k+1)
    coeffs[-1] = 1

    c, e = get_ellipse(a, b)

    def f(xs):
        return NPC.chebval( (xs-c)/e, coeffs )

    return f



def compute_jackson_coefficients(n):
    assert isinstance(n, int)
    assert n >= 0

    if n == 0:
        return NP.full_like(xs, 1)

    k = 1.0 * NP.arange(0, n+1)
    r = NP.pi / (n+1)
    cs = (1 - k/(n+1)) * NP.cos(k*r) + NP.sin(k*r) / ((n+1) * NP.tan(r))

    assert cs.size == n+1

    return cs



def heaviside(xs):
    if not isinstance(xs, NP.ndarray):
        return 0 if xs < 0 else 1

    t = xs >= 0
    ys = NP.empty_like(xs)
    ys[t] = 1
    ys[~t] = 0

    return ys



def fit_chebyshev_jackson_polynomial(k, f):
    assert isinstance(k, int)
    assert k >= 0

    xs = NP.linspace(-1, +1, 100 )
    ys = f(xs)

    cs = NPC.chebfit(xs, ys, k)
    gs = compute_jackson_coefficients(k)

    return cs * gs



def make_heaviside_approximation(k, a, b):
    assert isinstance(k, int)
    assert k >= 0

    is_odd = lambda n: (n%2) == 1

    c, e = get_ellipse(a, b)
    cs = fit_chebyshev_jackson_polynomial(k, heaviside)
    cs[2::2] = 0
    if is_odd(k):
        cs[-1] = 0

    def f(xs):
        return NPC.chebval((xs-c)/e, cs)

    return f



def cayley(xs, p):
    return 1/xs * (xs - p)



def print_stats(label, make_f, degrees, a, b, x2, x1, x0, m=100):
    assert isinstance(a, numbers.Real)
    assert isinstance(b, numbers.Real)
    assert a < b

    fmt = '{:24s} {:2d}  {:8.2e} {:8.2e}  {:8.2e}'

    for k in degrees:
        f = make_f(k, a, b)
        g = lambda x: f(1/x) / f(1/x0)

        print fmt.format( label, k, g(x2), g(x1), f(1/x2)/f(1/x1) )



def main(argv):
    lambda_1 = 0.05;
    lambda_c = 1.0;
    a = 10.0 * lambda_c;
    b = float('inf')

    left = lambda_1
    right = 30*lambda_c

    x0 = a
    xs = NP.linspace(left, right, num=1000)
    assert NP.all( xs > 0 )


    print_stats( \
        '1/x', lambda k, a, b: (lambda x: x),
        [1], 0, 1.0/a, lambda_1, lambda_c, a)
    print_stats( \
        'Chebyshev', make_chebyshev_polynomial,
        [2,3,6], 0, 1.0/a, lambda_1, lambda_c, a)
    print_stats( \
        'Chebyshev-Jackson', make_heaviside_approximation,
        [5,7], 0, 1.0/(2*a), lambda_1, lambda_c, a)
    print_stats( \
        'Chebyshev-Jackson(shift)', make_heaviside_approximation,
        [5,7], 0, 1.0/(2*a), lambda_1+lambda_c, lambda_c+lambda_c, a+lambda_c)


    # plot chebyshev polynomial damping
    chebyshev_label='Chebyshev({:d}, {:.1f}, {:.1f})'

    for k in [2,3,6]:
        f = make_chebyshev_polynomial(k, 1/b, 1/a)
        ys = f(1/xs) / f(1/x0)

        PP.plot(xs, abs(ys), label=chebyshev_label.format(k, a, b))


    # cayley
    cs = 50 * cayley(xs, a) / cayley(lambda_c, a)
    PP.plot(xs, abs(cs), 'k:', linewidth=2.0, label='Cayley')


    # 1/x
    PP.plot(xs, x0/xs, 'k', label='1/x')


    # heaviside + chebyshev-jackson polynomial damping
    fmt = 'Chebyshev-Jackson({:d})'

    for k in [3, 5, 7]:
        f = make_heaviside_approximation(k, 0, 1.0/(2*a))
        ys = f(1/xs) / f(1/x0)

        PP.plot( xs, abs(ys), '--', label=fmt.format(k) )


    # plot vertical lines
    PP.plot( (lambda_c,lambda_c), (0, 1e11), 'k-' )
    PP.plot( (a, a), (0, 1e11), 'k-' )


    # modify plot
    axes = PP.gca()
    axes.set_xlim([left, right])
    axes.set_ylim([1e-3, 1e6])
    PP.yscale('log')

    PP.legend()
    PP.grid()
    PP.show()



if __name__ == '__main__':
    sys.exit(main(sys.argv))
