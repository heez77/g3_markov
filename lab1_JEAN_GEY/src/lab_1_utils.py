import numpy as np
import scipy
from scipy import stats
import plotly.express as px
import pandas as pd

class exercise_1_workflow():
    """ A workflow of the first exercise. This exercise simulates a system of K = 30 particles (labeled from 1 to K) evolving in a closed box. 
    The box is divided into two compartments in contact with each other, respectively identified by an index, 0 and 1. 
    A hole at the interface between the two compartments allows the particles to move from one compartment to the other.

    Args:
        K (int): Number of particles in the system
    """
    def __init__(self, K:int):
        
        assert K>0, "K must be strictly positive"
        self.K = K

    def initialize_P(self):
        """Generate the P matrix of this Markov chain.
        """
        self.P = np.diag(np.arange(1/self.K,1+1/self.K,1/self.K), -1) + np.diag(np.arange(1,0, -1/self.K), 1)

    def check_invariant_distribution(self, distribution: stats._distn_infrastructure.rv_frozen)->bool:
        """This function checks if the distribution is invariant for our system.

        Args:
            distribution (scipy.stats._distn_infrastructure.rv_frozen): The distribution you want to test. Must be a scipy stats distribution.

        Returns:
            bool: Returns True if the distribution is invariant for the transition matrix of the problem.
        """
        self.pi = np.array([distribution.pmf(i) for i in range(self.K+1)])
        return np.count_nonzero(np.abs(self.pi @ self.P - self.pi) < np.full(shape=(self.K+1), fill_value=1e-9)) == self.K+1

    
    def ehrenfest(self, n_max:int, rng_seed=np.random.default_rng(420), show=True):
        """This function simulate a trajectory of this Markov Chain. It used the initial state given in the initialization. 
        You can display a figure of the trajectory if you want.

        Args:
            n_max (int): Number of steps for this trajectory.
            rng_seed (np.random._generator.Generator, optional): The rng seed used to determine the trajectory. Defaults to np.random.default_rng(420).
            show (bool, optional): If True displays the graph of the trajectory. Defaults to True.
        """
        assert n_max>0, "n_max must be strictly positive"

        X = np.zeros((n_max))
        for i in range(1, n_max):
            if rng_seed.random()< X[i-1]/self.K:
                X[i] = X[i-1] -1
            else:
                X[i] = X[i-1] + 1
        self.X = X
        self.fig = px.line(pd.DataFrame({"step":[i for i in range(n_max)], "Number of particles in box 0" : X}), x='step', y='Number of particles in box 0',
                            title="Evolution of the number of particles in box 0")
        if show:
            self.fig.show()
    
    def plot_two_histograms(self):
        """This function compares the histogram of the simulation and the histogram of the invariant probability.
        """
        
        df = pd.DataFrame({"step":[i for i in range(self.X.shape[0])], "État" : self.X})[['État']].value_counts(normalize=True).to_frame('Density')
        df.reset_index(inplace=True)
        df['Valeur'] = ['Histogramme simulation' for _ in range(len(df))]
        df = pd.concat([df, pd.DataFrame({'État': [i for i in range(self.pi.shape[0])], 
                                            'Density':self.pi, 
                                            'Valeur':['Histogramme probabilité invariante' for _ in range(self.pi.shape[0])]})])
        fig = px.bar(df, x='État', y='Density', color='Valeur', barmode='group',
                     title='Comparaison des histogrammes entre la simulation et la propabilité invariante')
        fig.show()
        
    def average_return_time(self, n_average:int=5, rng_seed=np.random.default_rng(420), print_result=True): 
        """This function prints the average return time to 0 for this Markov Chain using trajectories of 5 000 steps.

        Args:
            n_average (int, optional): Number of trajectories used to computes the average return time to 0. Defaults to 5.
            rng_seed (np.random._generator.Generator, optional): The rng seed used to determine the trajectory. Defaults to np.random.default_rng(420).
            print_result (bool, optional): If True prints the value of the average return time to 0. Defaults to True.
        """
        
        assert n_average>0, "n_average must be strictly positive"
        T = 0
        for _ in range(n_average):
            X = 0
            step = 0
            return_bool = False
            while not return_bool and step <= 100_000:
                step += 1
                if rng_seed.integers(0, self.K+1)<= X:
                    X = X -1
                else:
                    X = X + 1
                if X==0:
                    return_bool = True
                    
            if return_bool:
                T+= step
            else:
                n_average -= 1
        if n_average>0: 
            if print_result:     
                print(f"Le temps de retour moyen en 0 pour {self.K} particules est de : {T/n_average} steps.")
            return T/n_average
        else:
            print(f'Pour K = {self.K}, impossible de trouver un retour a 0.')
            return None

    def average_return_time_evolution(self, K_min:int=1, K_max:int=20, n_average:int=5):
        """This function plots the evolution of the average return time between K_min and K_max particles.

        Args:
            K_min (int, optional): Minimum number of particles. Defaults to 1.
            K_max (int, optional): Maximum number of particles. Defaults to 20.
            n_average (int, optional): Number of trajectories used to computes the average return time to 0. Defaults to 5.
        """
        assert n_average>0, "n_average must be strictly positive"
        old_k = self.K
        T_history = np.zeros((K_max + 1 - K_min), dtype='float')
        for i, k in enumerate(range(K_min, K_max+1)):
            self.K = k
            T_history[i] = self.average_return_time(n_average, print_result=False)
        self.K = old_k
        df = pd.DataFrame({"K":range(K_min, K_max+1), "Average return time": T_history})
        df.dropna(inplace=True)
        fig = px.line(df, x="K", y="Average return time",
                title= "Evolution du temps de retour moyen en 0 en fonction du nombre de particules K")
        fig.show()

class exercise_2_workflow():
    """A workflow of the second exercise of this notebook. This exercise simulates a system with 3 states, 
    a given P matrix and a given initial state.
    
    Args:
        mu_init (np.array): The initial state position.
        P_matrix (np.array): The P matrix for the given Matrix Chain.
    """
    def __init__(self, mu_init:np.array=np.array([0,1,0]), P_matrix:np.array=np.array([[0.2,0.7,0.1],
                                                                                        [0.9, 0, 0.1],
                                                                                        [0.2, 0.8, 0]])):
        self.mu_init = mu_init
        self.P = P_matrix
    
    def simulate_dthmc(self,n_max:int,  rng_seed:np.random._generator.Generator=np.random.default_rng(420), show:bool=True):
        """This function simulate a trajectory of this Markov Chain. It used the initial state given in the initialization. 
        You can display a figure of the trajectory if you want.

        Args:
            n_max (int): Number of steps for this trajectory.
            rng_seed (np.random._generator.Generator, optional): The rng seed used to determine the trajectory. Defaults to np.random.default_rng(420).
            show (bool, optional): If True displays the graph of the trajectory. Defaults to True.
        """
        X = np.zeros((n_max), dtype= 'int')
        X[0] = np.argmax(self.mu_init) + 1
        for i in range(1, n_max):
            X[i] = rng_seed.choice([1,2,3], p = self.P[X[i-1] - 1, :])
        self.X = X
        self.fig = px.histogram(pd.DataFrame({"step":[i for i in range(n_max)], "État" : X}), x='État',
                            title="Histogramme des états visités (normalisé)", histnorm='probability')
        if show:
            self.fig.show()


    def compute_invariant_distribution(self):
        """This function computes the invariant distribution for the given P matrix. First, the function checks if there is 
        an eigenvector associated with the eigenvalue 1. Then, it computes the normalized eigenvector which is the invariant 
        distribution.
        """
        valP, vecP = np.linalg.eig(self.P.T)
        idx = np.where(np.abs(valP-np.full(valP.shape, fill_value=1.))<=1e-9)[0]
        self.normalized_vector = None
        if len(idx)>1:
            print(f"La distribution invariante n'est pas unique, il y en a : {len(idx)}")
            self.normalized_vector = np.real(vecP[:,idx[0]] / np.sum(vecP[:,idx[0]]))
            print(f"La distribution invariante choisie est : {self.normalized_vector}")
        elif len(idx)==1:
            print(f"La distribution invariante est unique.")
            self.normalized_vector = np.real(vecP[:,idx[0]] / np.sum(vecP[:,idx[0]]))
            print(f"La distribution invariante est : {self.normalized_vector}")
        else:
            print("Il n'y a pas de distribution invariante.")

    def compute_probability_distribuction(self, n_max:int):
        """This function computes the probability distribution for each steps in the trajectory. 

        Args:
            n_max (int): Number of steps in the trajectory.
        """
        mu_history = np.zeros((n_max, self.mu_init.shape[0]), dtype='float')
        mu_history[0] = self.mu_init
        for i in range(1, n_max):
            mu_history[i] = mu_history[i-1] @self.P
        self.mu_history =  mu_history

    def check_invariant_distribution(self):
        """This function checks if the probability distribution of the last point in the trajectory si almost equals to the 
        invariant distribution.
        """
        equality_check = np.count_nonzero(np.abs(self.mu_history[-1] - self.normalized_vector) < np.full(shape=self.normalized_vector.shape, fill_value=1e-9)) == self.normalized_vector.shape[0]
        if equality_check:
            print(f"La distribution de X(n) tend vers la distribution invariante obtenue à la question 5 : {self.normalized_vector}")
        else:
            print(f"La distribution de X(n) ne tend pas vers la distribution invariante obtenue à la question 5 : {self.normalized_vector}")
    
    def plot_two_histograms(self):
        """This function compares the histogram of the simulation and the histogram of the invariant probability.
        """
        df = pd.DataFrame({"step":[i for i in range(self.X.shape[0])], "État" : self.X})[['État']].value_counts(normalize=True).to_frame('Density')
        df.reset_index(inplace=True)
        df['Valeur'] = ['Histogramme simulation' for _ in range(len(df))]
        df = pd.concat([df, pd.DataFrame({'État': [i+1 for i in range(self.mu_init.shape[0])], 
                                            'Density':self.normalized_vector, 
                                            'Valeur':['Histogramme probabilité invariante' for _ in range(self.mu_init.shape[0])]})])
        fig = px.bar(df, x='État', y='Density', color='Valeur', barmode='group',
                     title='Comparaison des histogrammes entre la simulation et la propabilité invariante')
        fig.show()
        
    def plot_mu_evolution(self):
        """This function plots the evolution of the probability distribution.
        """
        
        fig = px.line(pd.DataFrame(self.mu_history, columns=['Distribution en 0', 'Distribution en 1', 'Distribution en 2']),
                      title='Evolution des composantes de la distribution')
        fig.update_xaxes(dict(title='steps'))
        fig.show()
        
    def plot_norm_difference(self):
        """This function plots the difference using L-norm between the invariant distribution and the probability distribution.
        """
        
        pi_history = np.tile(self.normalized_vector, (self.mu_history.shape[0],1))
        norm_history = np.linalg.norm(self.mu_history - pi_history,ord=1,axis=1)
        fig = px.line(pd.DataFrame(norm_history, columns=['||mu-pi||']), title = "Évolution de la différence en la distibution et pi en fonction des steps")
        fig.update_xaxes(dict(title='steps'))
        fig.show()
        
    def theorical_return_time(self):
        """This function computes the theoretical average return time on each step. This is just the inverse of the invariant probability.
        """
        theorical_return_time = 1/self.normalized_vector
        
        for i in range(theorical_return_time.shape[0]):
            print(f"Moyenne du temps de retour en {i+1} : {round(theorical_return_time[i],2)}")
        
        
        

    def empirical_average_return_time(self, rng_seed:np.random._generator.Generator=np.random.default_rng(420), n_trajectories:int = 100):
        """This function computes the average return time on each step (1, 2, 3). A random generator is used to replicate 
        same trajectories.

        Args:
            rng_seed (np.random._generator.Generator, optional): The rng seed used to determine the trajectory. Defaults to np.random.default_rng(420).
            n_trajectories (int, optional): Number of trajectories to calculate the average return time. Defaults to 100.
        """
        
        assert n_trajectories>0, "n_trajectories must be strictly positive"
        
        empirical_return_time = np.zeros((self.mu_init.shape[0]), dtype='float')
        
        for i in range(self.mu_init.shape[0]):
            return_time = 0
            for _ in range(n_trajectories):
                step = 0
                return_bool = False
                X = i
                while not return_bool:
                    step +=1
                    X = rng_seed.choice([i for i in range(self.mu_init.shape[0])], p = self.P[X, :])
                    return_bool = X == i
                return_time += step
            empirical_return_time[i] = return_time / n_trajectories
            
        for i in range(self.mu_init.shape[0]):
            print(f"Moyenne du temps de retour en {i+1} : {round(empirical_return_time[i],2)}")
        
    