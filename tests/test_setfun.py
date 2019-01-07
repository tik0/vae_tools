import unittest
from vae_tools import setfun

class SetfunTest(unittest.TestCase):

    def testConstruction(self):
        target = [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]
        output = setfun.powerset([1,2,3], minimum_elements_per_set = 0)
        self.assertListEqual(output, target)
        
        target = [(1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]
        output = setfun.powerset([1,2,3])
        self.assertListEqual(output, target)
        
        target = [[1,2], [1,3], [2,3], [1,2,3]]
        output = setfun.powerset([1,2,3], minimum_elements_per_set = 2, sets_as_list = True)
        self.assertListEqual(output, target)
        
        target_subset = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        target_superset = [3, 4, 3, 5, 4, 5, 6, 6, 6, 6, 6, 6]
        output_subset, output_superset = setfun.find_proper_subsets( setfun.powerset("abc"), cardinality_difference = 1)
        self.assertListEqual(output_subset, target_subset)
        self.assertListEqual(output_superset, target_superset)

if __name__ == "__main__": 
    unittest.main()