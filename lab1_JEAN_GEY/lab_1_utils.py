import numpy as np
import scipy
from scipy import stats
import plotly.express as px
import pandas as pd

class exercise_1_workflow():
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

    def check_invariant_distribution(self, distribution: stats._distn_infrastructure.rv_frozen)->bool:
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



class exercise_2_worflow():
    def __init__(self):
        self.mu_init = np.array([0,1,0])
        self.P = np.array([[0.2,0.7,0.1],
                            [0.9, 0, 0.1],
                            [0.2, 0.8, 0]])
    
    def simulate_dthmc(self,n_max,  rng_seed=np.random.default_rng(420), show=True):
        X = np.zeros((n_max), dtype= 'int')
        X[0] = np.argmax(self.mu_init)
        for i in range(1, n_max):
            X[i] = rng_seed.choice([0,1,2], p = self.P[X[i-1], :])
        self.fig = px.histogram(pd.DataFrame({"step":[i for i in range(n_max)], "État" : X}), x='État',
                            title="Histogramme des états visités (normalisé) dfezf", histnorm='probability')
        if show:
            self.fig.show()


    def compute_invariant_distribution(self):
        valP, vecP = np.linalg.eig(self.P.T)
        idx = np.where(np.abs(valP-np.full(valP.shape, fill_value=1.))<=1e-9)[0]
        self.normalized_vector = None
        if len(idx)>1:
            print("La distribution invariante n'est pas unique.")
        elif len(idx)==1:
            print("La distribution invariante est unique.")
        else:
            print("Il n'y a pas de distribution invariante.")

        if len(idx)>0:
            self.normalized_vector = np.real(vecP[:,idx[0]] / np.sum(vecP[:,idx[0]]))
            print(f"La distribution invariante est : {self.normalized_vector}")


    def compute_probability_distribuction(self, n_max:int):
        mu_history = np.zeros((n_max, self.mu_init.shape[0]), dtype='float')
        mu_history[0] = self.mu_init
        for i in range(1, n_max):
            mu_history[i] = mu_history[i-1] @self.P
        self.mu_history =  mu_history

    def check_invariant_distribution(self):
        equality_check = np.count_nonzero(np.abs(self.mu_history[-1] - self.normalized_vector) < np.full(shape=self.normalized_vector.shape, fill_value=1e-9)) == self.normalized_vector.shape[0]
        if equality_check:
            print(f"La distribution de X(n) tend vers la distribution invariante obtenue à la question 5 : {self.normalized_vector}")
        else:
            print(f"La distribution de X(n) ne tend pas vers la distribution invariante obtenue à la question 5 : {self.normalized_vector}")