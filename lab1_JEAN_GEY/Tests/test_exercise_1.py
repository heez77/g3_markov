import unittest
from src.lab_1_utils import exercise_1_workflow
import numpy as np
from scipy.stats import binom



class Test_Exercise_1(unittest.TestCase):
    def __init__(self):
        super().__init__(self)
        self.rng_seed = np.random.default_rng(420)
        self.workflow = exercise_1_workflow(self.rng_seed.integers(1, 50))
        
    def test_P_transition_matrix(self):
        self.workflow.initialize_P()
        self.assertTrue(False not in (np.abs(np.sum(self.workflow.P, axis=1)-np.full(self.workflow.P.shape[0], 1)<= np.full(self.workflow.P.shape[0], 1e-9))))

    def test_invariant_distribution_True(self):
        self.assertTrue(self.workflow.check_invariant_distribution(binom(self.K, 0.5)))
    
    def test_invariant_distribution_False(self):
        self.assertFalse(self.workflow.check_invariant_distribution(binom(self.K, 0.2)))

