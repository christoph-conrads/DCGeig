#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.



class Options:
    def __init__(self):
        self.n_direct = 1024
        self.internal_tol = NP.finfo(NP.float32).eps
        self.c_s = 10
        self.n_s_min = 32
        self.max_num_iterations = 10
