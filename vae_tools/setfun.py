#!/usr/bin/python

from itertools import chain, combinations

def powerset(iterable, minimum_elements_per_set = 1, sets_as_list = False):
    """Returns the powerset as list with sets as tuples 
    
    powerset([1,2,3], minimum_elements_per_set = 0) --> [() (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)]
    powerset([1,2,3]) --> [(1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)]
    powerset([1,2,3], minimum_elements_per_set = 2, sets_as_list = True) --> [[1,2] [1,3] [2,3] [1,2,3]]
    """
    s = list(iterable)
    powerset = list(chain.from_iterable(combinations(s, r) for r in range(minimum_elements_per_set, len(s)+1)))
    if sets_as_list:
        powerset = [list(current_set) for current_set in powerset]
    return powerset

def find_proper_subsets(powerset, cardinality_difference = 1, debug = False):
    """Returns the indecees of proper subsets A and supersets B with the constraint |A|=|B|-cardinality_difference
    
    We always assume ordered powersets as produced by powerset() (i.e. powerset("abc") -> [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c'), ('b', 'c'), ('a', 'b', 'c')])
    this gives us the indecees
    subset_idx: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    superset_idx [3, 4, 3, 5, 4, 5, 6, 6, 6, 6, 6, 6]
    Test with find_proper_subsets(powerset("abc"), debug = True)
    """
    subset_idx = [] # Which will be set A of the powerset
    superset_idx = [] # Which will be set B of the powerset

    for A, a_idx in zip(powerset, list(range(0, len(powerset)))):
        # A_is_proper_subset_of_B = True
        for B, b_idx in zip(powerset[a_idx:], list(range(a_idx, len(powerset)))):
            if len(A) is not len(B)-cardinality_difference:
                continue
            else: # Check every element
                for a in A:
                    A_is_proper_subset_of_B = True
                    found_a_in_b = False
                    for b in B:
                        if a is b:
                            found_a_in_b = True
                            break
                    if found_a_in_b is False:
                        A_is_proper_subset_of_B = False
                    if A_is_proper_subset_of_B:
                        if debug:
                            print("A:", A, " is proper subset of B:", B)
                        subset_idx.append(a_idx)
                        superset_idx.append(b_idx)
        # return the indecees for the corresponding sets
    #print("subset_idx: ", subset_idx)
    #print("superset_idx: ", superset_idx)
    return subset_idx, superset_idx
