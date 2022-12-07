import numpy as np
import scipy
import plotly.express as px
import pandas as pd

class exercice_1_workflow():
    def __init__(self, K:int):
        """of K = 30 particles (labeled from 1 to K) evolving in a closed box. 
        The box is divided into two compartments in contact with each other, respectively identified by an index, 0 and 1. 
        A hole at the interface between the two compartments allows the particles to move from one compartment to the other.

        Args:
            K (int): Number of particles in the system
        """
        self.K = K

    def initialize_P(self):
        """Generate P matrix for this problem.
        """
        P = np.zeros((self.K+1, self.K+1), dtype='float')
        P[0,1] = 1
        P[self.K, self.K-1] = 1
        for i in range(1, self.K):
            P[i,i+1] = (self.K-i)/self.K
            P[i,i-1] = i/self.K
        self.P = P

    def check_invariant_distribution(self, distribution: scipy.stats._distn_infrastructure.rv_frozen)->bool:
        """Check if the distribution is invariant for our problem.

        Args:
            distribution (scipy.stats._distn_infrastructure.rv_frozen): The distribution you want to test. Must be a scipy stats distribution.

        Returns:
            bool: Returns True if the distribution is invariant for the transition matrix of the problem.
        """
        self.pi = np.array([distribution.pmf(i) for i in range(self.K+1)])
        return np.count_nonzero(np.abs(self.pi @ self.P - self.pi) < np.full(shape=(self.K+1), fill_value=1e-9)) == self.K+1

    
    def ehrenfest(self, n_max:int, rng_seed=np.random.default_rng(420), show=True):
        assert n_max>0, "n_max must be strictly positive"

        X = np.zeros((n_max))
        self.T = None
        for i in range(1, n_max):
            if rng_seed.integers(0, self.K+1)<= X[i-1]:
                X[i] = X[i-1] -1
            else:
                X[i] = X[i-1] + 1

            if X[i]==0 and self.T is None:
                self.T = i
        self.fig = px.line(pd.DataFrame({"step":[i for i in range(n_max)], "Number of particles in box 0" : X}), x='step', y='Number of particles in box 0',
                            title="Evolution of the number of particles in box 0")
        if show:
            self.fig.show()