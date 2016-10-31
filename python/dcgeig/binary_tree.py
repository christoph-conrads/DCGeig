#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numbers

import copy


class BinaryTree:
    def __init__(self, left_child, right_child, data):
        if not isinstance(data, dict):
            raise TypeError('data must be a dictionary')

        # this test does not catch strings that are not identifiers like '1'
        if filter(lambda k: not isinstance(k, str), data.keys()):
            raise ValueError('data keys must be strings')

        if 'left_child' in data or 'right_child' in data:
            raise AttributeError('Illegal keys found')

        self.left_child = left_child
        self.right_child = right_child
        self.__dict__.update(data)


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



def make_leaf_node(data):
    return BinaryTree(None, None, data)



def make_internal_node(left_child, right_child, data):
    return BinaryTree(left_child, right_child, data)



def add_postorder_id(tree, sid=1):
    assert isinstance(tree, BinaryTree)
    assert not hasattr(tree, 'id')

    if Tree.is_leaf_node(tree):
        new_tree = copy.copy(tree)
        new_tree.id = sid
        return new_tree

    new_left = add_postorder_id(tree.left_child, sid)
    new_right= add_postorder_id(tree.right_child, new_left.id+1)

    new_tree = copy.copy(tree)
    new_tree.left_child = new_left
    new_tree.right_child = new_right
    new_tree.id = new_right.id + 1

    return new_tree
