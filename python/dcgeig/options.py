#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

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
