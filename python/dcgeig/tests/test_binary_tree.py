#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import dcgeig.binary_tree as binary_tree

import unittest



class Test_Node(unittest.TestCase):
    def test_simple(self):
        data = {'a': 1, 'b': 2.0, 'c': 'd'}
        tree = binary_tree.make_leaf_node(0, data)

        self.assertEqual( tree.a, data['a'] )
        self.assertEqual( tree.b, data['b'] )
        self.assertEqual( tree.c, data['c'] )


    def test_invalid(self):
        data = {1: '2'}

        with self.assertRaises(ValueError):
            tree = binary_tree.make_leaf_node(0, data)

        with self.assertRaises(ValueError):
            leaf = binary_tree.make_leaf_node(0)
            tree = binary_tree.make_internal_node(leaf, leaf, 0, data)



class Test_add_postorder_id(unittest.TestCase):
    def check_tree(self, tree):
        self.assertTrue( isinstance(tree.id, int) )

        if tree.is_leaf_node():
            return

        self.assertTrue( hasattr(tree, 'left_child') )
        self.assertTrue( hasattr(tree, 'right_child') )

        left = tree.left_child
        right = tree.right_child

        self.assertTrue( isinstance(left, binary_tree.Node) )
        self.assertTrue( isinstance(right, binary_tree.Node) )

        self.assertTrue( left.id < right.id )
        self.assertEqual( right.id + 1, tree.id )

        self.check_tree(tree.left_child)
        self.check_tree(tree.right_child)


    def test_simple(self):
        left = binary_tree.make_leaf_node(0)
        right = binary_tree.make_leaf_node(0)
        root = binary_tree.make_internal_node(left, right, 0)

        id_tree = binary_tree.add_postorder_id(root)
        self.check_tree( id_tree )


    def test_unbalanced(self):
        t0 = binary_tree.make_leaf_node(0)
        t1 = binary_tree.make_leaf_node(0)
        t2 = binary_tree.make_internal_node(t0, t1, 0)
        t3 = binary_tree.make_leaf_node(0)
        t4 = binary_tree.make_internal_node(t2, t3, 0)

        id_tree = binary_tree.add_postorder_id(t4)
        self.check_tree(id_tree)



if __name__ == '__main__':
    unittest.main()
