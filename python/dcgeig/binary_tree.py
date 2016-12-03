#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numbers

import copy


class Node:
    def __init__(self, left_child, right_child, n, data):
        assert isinstance(n, int)
        assert n >= 0
        assert isinstance(data, dict)
        assert 'left_child' not in data
        assert 'right_child' not in data
        assert 'n' not in data

        if filter(lambda key: not isinstance(key, str), data.keys()):
            raise ValueError('data keys must be strings')

        self.__dict__.update(data)
        self.left_child = left_child
        self.right_child = right_child
        self.n = n


    def has_left_child(self):
        return self.left_child is not None

    def has_right_child(self):
        return self.right_child is not None

    def is_leaf_node(self):
        return (not self.left_child) and (not self.right_child)


    def get_height(self):
        if self.is_leaf_node():
            return 0

        assert self.has_left_child()
        assert self.has_right_child()

        left = self.left_child
        right = self.right_child

        return max(left.get_height(), right.get_height()) + 1



def make_leaf_node(n, data={}):
    return Node(None, None, n, data)



def make_internal_node(left_child, right_child, n, data={}):
    return Node(left_child, right_child, n, data)



def add_postorder_id(node, sid=1):
    assert isinstance(node, Node)
    assert not hasattr(node, 'id')

    if node.is_leaf_node():
        new_node = copy.copy(node)
        new_node.id = sid
        return new_node

    new_left = add_postorder_id(node.left_child, sid)
    new_right= add_postorder_id(node.right_child, new_left.id+1)

    new_node = copy.copy(node)
    new_node.left_child = new_left
    new_node.right_child = new_right
    new_node.id = new_right.id + 1

    return new_node
