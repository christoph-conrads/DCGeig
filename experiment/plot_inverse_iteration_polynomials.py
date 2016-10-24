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

    alpha = NP.pi / (n + 2)
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



def cayley(xs, p):
    return 1/xs * (xs - p)



def main(argv):
    lambda_1 = 0.05;
    lambda_c = 1.0;
    a = 10.0 * lambda_c;
    b = NP.sqrt(2) * 10.0 * lambda_c;
    c, e = get_ellipse(1/b, 1/a)

    left = lambda_1
    right = 30*lambda_c

    x0 = a
    xs = NP.linspace(left, right, num=1000)
    assert NP.all( xs > 0 )

    chebyshev_label='Chebyshev({:d}, {:.1f}, {:.1f})'


    # plot chebyshev polynomial damping
    for k in range(2, 4):
        f = make_chebyshev_polynomial(k, 1/b, 1/a)
        ys = f(1/xs) / f(1/x0)

        PP.plot(xs, abs(ys), label=chebyshev_label.format(k, a, b))


    # cayley
    cs = 50 * cayley(xs, a) / cayley(lambda_c, a)
    PP.plot(xs, abs(cs), 'k:', linewidth=2.0, label='Cayley')

    # 1/x
    PP.plot(xs, x0/xs, 'k', label='1/x')

    # damp interval [1/infty, 1/a]
    k = 2
    inf = float('inf')
    f_inf = make_chebyshev_polynomial(k, 0, 1/a)
    ys = f_inf(1/xs) / f_inf(1/x0)
    PP.plot( xs, abs(ys), label=chebyshev_label.format(k, a, inf) )


    # heaviside + chebyshev-jackson polynomial damping
    fmt = 'Chebyshev-Jackson({:d})'
    c_h, e_h = get_ellipse(0, 1.0/(2*a))

    for k in [3, 7]:
        cs = fit_chebyshev_jackson_polynomial(k, heaviside)
        cs[2::2] = 0
        degree = k if (k%2 == 0) else k-1
        cs = cs[:degree+1]

        f = lambda xs: NPC.chebval( (xs-c_h)/e_h, cs )
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
