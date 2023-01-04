import numpy as np
import matplotlib.pyplot as plt



class exercise_1_workflow():
    """ A workflow of the first exercise. bla-bla-bla
        Args:
            rng (int): random seed (or random generator).
    """

    def __init__(self, rng=np.random.default_rng(420)):
        self.rng = rng
        self.X_mm1inf = None
        self.T_mm1inf = None


    def run_mm1inf(self, lambd: float, mu: float, niter: int):
        """
        Args:
            lambd (float): birth rate.
            mu (float): death rate
            
            niter (int): number of changes (events) in 
                        the process.
        Raises:
            ValueError: error triggered if lambd <= 0.
            ValueError: error triggered if mu <= 0.
        Returns:
            X (array_like): trajectory (X(t_n)).
            T (array_like): time instants at which a change in 
                            the process occurs (t_n).
        """

        assert lambd >0, 'lambda must be strictly positive'
        assert mu>0, 'mu must be strictly positive'

        self.lambd_mm1inf = lambd
        self.mu_mm1inf = mu

        X = np.zeros(niter)
        T = np.zeros(niter)

        for i in range(1, niter):
            birth_time = self.rng.exponential(1/lambd)
            death_time = self.rng.exponential(1/mu)
            if X[i-1]==0 or birth_time< death_time:
                T[i] = T[i-1] + birth_time
                X[i] = X[i-1] + 1 
            else:
                T[i] = T[i-1] + death_time
                X[i] = X[i-1] - 1
        self.X_mm1inf = X
        self.T_mm1inf = T

    def plot_step_mm1inf(self):
        assert self.X_mm1inf is  not None and self.T_mm1inf is not None, "You didn't done any simulation"
        plt.step(self.T_mm1inf[-100:],self.X_mm1inf[-100:])
        plt.title(f"Evolution of the M/M/1/inf process with lambda={self.lambd_mm1inf} and mu={self.mu_mm1inf}")
        plt.xlabel("Time")
        plt.ylabel("Number of people in the process")
    
    def plot_two_histograms_mm1inf(self):
        max_people_in_queue = int(np.amax(self.X_mm1inf))
        self.rho_mm1inf = self.lambd_mm1inf/self.mu_mm1inf
        pi = np.zeros(max_people_in_queue)
        for i in range(max_people_in_queue):
            pi[i] = (1- self.rho_mm1inf)* self.rho_mm1inf**i

        counts, bins = np.histogram(self.X_mm1inf[:-1], max_people_in_queue, density = True, weights=np.diff(self.T_mm1inf))

        width = 0.25
        plt.bar(bins[:-1] + width, counts, width = width, label='X')
        plt.bar(bins[:-1], pi, width = width, label = 'invariant distrib')
        plt.xticks(bins[:-1] + width/2,bins[:-1])
        plt.title("")
        plt.legend()
        plt.show()


    def run_mm1K(self, lambd: float, mu: float, K: int, niter: int):
        """
        Args:
            lambd (float): birth rate.
            mu (float): death rate
            rng (int): random seed (or random generator).
            niter (int): number of changes (events) in 
                        the process.
        Raises:
            ValueError: error triggered if lambd <= 0.
            ValueError: error triggered if mu <= 0.
        Returns:
            X (array_like): trajectory (X(t_n)).
            T (array_like): time instants at which a change in 
                            the process occurs (t_n).
        """

        assert lambd >0, 'lambda must be strictly positive'
        assert mu>0, 'mu must be strictly positive'

        self.lambd_mm1K = lambd
        self.mu_mm1K = mu
        self.K = K
        X = np.zeros(niter)
        T = np.zeros(niter)

        for i in range(1, niter):
            birth_time = self.rng.exponential(1/lambd)
            death_time = self.rng.exponential(1/mu)
            if X[i-1]==0 or (birth_time< death_time and X[i-1]<K):
                T[i] = T[i-1] + birth_time
                X[i] = X[i-1] + 1 
            else:
                T[i] = T[i-1] + death_time
                X[i] = X[i-1] - 1
        self.X_mm1K = X
        self.T_mm1K = T

    
    def plot_step_mm1K(self):
        assert self.X_mm1K is  not None and self.T_mm1K is not None, "You didn't done any simulation"
        plt.step(self.T_mm1K[-100:],self.X_mm1K[-100:])
        plt.title(f"Evolution of the M/M/1/inf process with lambda={self.lambd_mm1inf} and mu={self.mu_mm1inf}")
        plt.xlabel("Time")
        plt.ylabel("Number of people in the process")

    
    def plot_two_histograms_mm1K(self):
        max_people_in_queue = int(np.amax(self.X_mm1K))
        self.rho_mm1K = self.lambd_mm1K/self.mu_mm1K
        pi = np.zeros(self.K+1)
        for i in range(self.K+1):
            pi[i] = ((1- self.rho_mm1K)* self.rho_mm1K**i) / (1-self.rho_mm1K**(self.K+1))

        counts, bins, _ = plt.hist(self.X_mm1K[:-1], range(self.K+2), density = True, weights=np.diff(self.T_mm1K), label = 'Empirical histogram')
        plt.bar([i +0.5 for i in range(self.K+1)], pi,alpha = 0.3, label = 'invariant distrib', color='r')
        plt.title("")
        plt.legend()
        plt.show()