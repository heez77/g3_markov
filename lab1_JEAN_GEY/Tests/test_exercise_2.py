import unittest
from src.lab_1_utils import exercise_2_workflow
import numpy as np



class Test_Exercise_2(unittest.TestCase):
    def __init__(self, name):
        unittest.TestCase.__init__(self, name)
        self.rng_seed = np.random.default_rng(420)
        self.mu_init = np.zeros(3)
        self.mu_init[self.rng_seed.integers(0,3)] = 1
        self.workflow = exercise_2_workflow(mu_init=self.mu_init)
        self.workflow.compute_invariant_distribution()
    
    def test_compute_invariant_distribution_invariant(self):
        self.assertTrue(False not in (np.abs(self.workflow.normalized_vector @ self.workflow.P - self.workflow.normalized_vector)<= np.full(self.workflow.normalized_vector.shape[0], 1e-9)))
        
    
    def test_compute_invariant_distribution_normalized(self):
        self.assertTrue(np.abs(np.linalg.norm(self.workflow.normalized_vector, 1) - 1 ) <= 1e-9)