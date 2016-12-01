#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.


import numpy as NP



def nop(_):
    pass


class Options(object):
    def __init__(self):
        self.n_direct = 1000
        self.n_s_min = 50

        self.num_trial_vectors = 50
        self.polynomial_degree = 50

        self.eta_max = NP.finfo(NP.float32).eps
        self.delta_max = 1e0

        self.show = nop

        assert self.num_trial_vectors <= self.n_direct
