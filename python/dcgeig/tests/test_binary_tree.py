#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import dcgeig.binary_tree as binary_tree

import unittest



class Test_BinaryTree(unittest.TestCase):
    def test_simple(self):
        data = {'a': 1, 'b': 2.0, 'c': 'd'}
        tree = binary_tree.make_leaf_node(data)

        self.assertEqual( tree.a, data['a'] )
        self.assertEqual( tree.b, data['b'] )
        self.assertEqual( tree.c, data['c'] )


    def test_invalid(self):
        data = {1: '2'}

        with self.assertRaises(ValueError):
            tree = binary_tree.make_leaf_node(data)

        with self.assertRaises(ValueError):
            leaf = binary_tree.make_leaf_node({})
            tree = binary_tree.make_internal_node(leaf, leaf, data)



class Test_add_postorder_id(unittest.TestCase):
    def check_tree(self, tree):
        self.assertTrue( isinstance(tree.id, int) )

        if tree.is_leaf_node():
            return

        self.assertTrue( hasattr(tree, 'left_child') )
        self.assertTrue( hasattr(tree, 'right_child') )

        left = tree.left_child
        right = tree.right_child

        self.assertTrue( isinstance(left, binary_tree.BinaryTree) )
        self.assertTrue( isinstance(right, binary_tree.BinaryTree) )

        self.assertTrue( left.id < right.id )
        self.assertEqual( right.id + 1, tree.id )

        self.check_tree(tree.left_child)
        self.check_tree(tree.right_child)


    def test_simple(self):
        left = binary_tree.make_leaf_node({})
        right = binary_tree.make_leaf_node({})
        root = binary_tree.make_internal_node(left, right, {})

        id_tree = binary_tree.add_postorder_id(root)
        self.check_tree( id_tree )


    def test_unbalanced(self):
        t0 = binary_tree.make_leaf_node({})
        t1 = binary_tree.make_leaf_node({})
        t2 = binary_tree.make_internal_node(t0, t1, {})
        t3 = binary_tree.make_leaf_node({})
        t4 = binary_tree.make_internal_node(t2, t3, {})

        id_tree = binary_tree.add_postorder_id(t4)
        self.check_tree(id_tree)



if __name__ == '__main__':
    unittest.main()
