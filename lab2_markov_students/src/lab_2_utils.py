import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import animation, rc
from IPython.display import HTML
import os


class exercise_1_workflow():
    """ A workflow of the first exercise. The purpose of this exercise is to simulate M/M/1/inf and M/M/1/K queues 
        and compare simulations with theorical values.

        Args:
            rng (int): random seed (or random generator).
    """

    def __init__(self, rng=np.random.default_rng(420)):
        self.rng = rng
        self.X_mm1inf = None
        self.T_mm1inf = None
        self.X_mm1K = None
        self.T_mm1K = None


    def run_mm1inf(self, lambd: float, mu: float, niter: int):
        """
        This function simulate an M/M/1/inf queue with given birth/death ratesand the number of iterations.
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

        assert lambd >0, 'lambda should be strictly positive'
        assert mu>0, 'mu should be strictly positive'

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
        """ 
        This function displays the generated trajectory of the process (M/M/1/inf).
        """

        assert self.X_mm1inf is  not None and self.T_mm1inf is not None, "You didn't done any simulation"
        plt.step(self.T_mm1inf[-100:],self.X_mm1inf[-100:])
        plt.title(f"Evolution of the M/M/1/inf process with lambda={self.lambd_mm1inf} and mu={self.mu_mm1inf}")
        plt.xlabel("Time")
        plt.ylabel("Number of people in the process")
    
    def plot_two_histograms_mm1inf(self):
        """ 
        This function displays the normalized histogram of trajectory and it's stationary distribution for M/M/1/inf queue.
        """

        max_people_in_queue = int(np.amax(self.X_mm1inf))
        self.rho_mm1inf = self.lambd_mm1inf/self.mu_mm1inf
        pi = np.zeros(max_people_in_queue)
        for i in range(max_people_in_queue):
            pi[i] = (1- self.rho_mm1inf)* self.rho_mm1inf**i

        counts, bins = np.histogram(self.X_mm1inf[:-1], max_people_in_queue, density = True, weights=np.diff(self.T_mm1inf)) #Ponderate by the time interval

        width = 0.25
        plt.bar(bins[:-1] + width, counts, width = width, label='X')
        plt.bar(bins[:-1], pi, width = width, label = 'invariant distrib')
        plt.xticks(bins[:-1] + width/2,bins[:-1])
        plt.title("")
        plt.legend()
        plt.show()

    def compare_average_theorical_distrib_inf(self):
        """
        This function computes the relative error rate betwen the invariant distribution and the normalized histogram of the trajectory.
        """
        average = np.average(self.X_mm1inf[:-1], weights=np.diff(self.T_mm1inf)) #Ponderate by the time interval
        theorical_value = self.rho_mm1inf / (1-self.rho_mm1inf)
        print(f"the average number of customers is : {average}.")
        print(f"The theorical value is : {theorical_value}.")
        print(f"We obtain a relative difference of {np.abs((average-theorical_value)/theorical_value)}")

    def run_mm1K(self, lambd: float, mu: float, K: int, niter: int):
        """
        This function simulate an M/M/1/inf queue with given birth/death rates, the number of iterations and the maximum of people in the queue.
        Args:
            lambd (float): birth rate.
            mu (float): death rate
            K (int): maximum number of peaople in the process.
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

        assert lambd >0, 'lambda should be strictly positive'
        assert mu>0, 'mu should be strictly positive'

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
        """ 
        This function displays the generated trajectory of the process (M/M/1/K).
        """
        assert self.X_mm1K is  not None and self.T_mm1K is not None, "You didn't done any simulation."
        plt.step(self.T_mm1K[-100:],self.X_mm1K[-100:])
        plt.title(f"Evolution of the M/M/1/K process with lambda={self.lambd_mm1inf} and mu={self.mu_mm1inf}")
        plt.xlabel("Time")
        plt.ylabel("Number of people in the process")

    
    def plot_two_histograms_mm1K(self):
        """ 
        This function displays the normalized histogram of trajectory and it's stationary distribution for M/M/1/K queue.
        """
        self.rho_mm1K = self.lambd_mm1K/self.mu_mm1K
        pi = np.zeros(self.K+1)
        for i in range(self.K+1):
            pi[i] = ((1- self.rho_mm1K)* self.rho_mm1K**i) / (1-self.rho_mm1K**(self.K+1))
        self.pi = pi
        plt.hist(self.X_mm1K[:-1], range(self.K+2), density = True, weights=np.diff(self.T_mm1K), label = 'Empirical histogram', color='b')
        plt.bar([i +0.5 for i in range(self.K+1)], pi,alpha = 0.3, label = 'invariant distrib', color='r')
        plt.title("")
        plt.legend()
        plt.show()
    
    def compare_average_theorical_distrib_K(self):
        """
        This function computes the relative error rate betwen the invariant distribution and the normalized histogram of the trajectory for M/M/1/K queue.
        """
        average = np.average(self.X_mm1K[:-1], weights=np.diff(self.T_mm1K))
        theorical_value = np.sum([i*p for i, p in enumerate(list(self.pi))])
        print(f"the average number of customers is : {average}.")
        print(f"The theorical value is : {theorical_value}.")
        print(f"We obtain a relative difference of {np.abs((average-theorical_value)/theorical_value)}")




class exercise_2_workflow():
    """ A workflow of the second exercise. The purpose of this exercise is to implement a Metropolis-Hastings algorithm to draw samples 
        from the 2D Ising model.

        Args:
            rng (int): random seed (or random generator).
    """

    def __init__(self, rng=np.random.default_rng(420)):
        self.rng = rng
    
    def metro_hasting_ising_2D(self, N:int,beta:float,num_samples:int, interval:int=20):
        """
        This function simulate a Metropolis-Hastings algorithm from the 2D ising model randomly initialized.

        Args:
            N (int) : grid dimension (N x N).
            beta (float) : Inverse of temperature.
            num_samples (int) : Number of iterations on the grid.
            interval (int): Interval between two screenshot of the simulation.
        
        Raises:
            ValueError: error triggered if N <= 0.
            ValueError: error triggered if num_samples <= 0.

        Returns:
            video(HTML) : A video of the simulation.

        """
        assert N>0, print("N should be positive.")
        assert num_samples>0, print("num_samples should be positive.")

        rng = self.rng
        fig = plt.figure()
        plt.grid(False)
        grid = 2*rng.integers(2, size=(N,N))-1
        imgs = [[plt.imshow(grid, cmap="gray", vmin=-1, vmax=1)]]
        for i in range(1, num_samples):
            x = rng.integers(N, size=1)[0]
            y = rng.integers(N, size=1)[0]
            delta_E = 2 * grid[x,y] * (grid[(x+1)%N,y] + grid[(x-1)%N,y] + grid[x,(y+1)%N] + grid[x,(y-1)%N])
            if rng.random() < np.exp(-beta * delta_E):
                grid[x,y] = -grid[x,y]
            if i % interval==0 :
                imgs.append([plt.imshow(grid, cmap="gray", vmin=-1, vmax=1)])
        

        
        anim = animation.ArtistAnimation(fig,imgs, blit=True)
        video = HTML(anim.to_html5_video())
        return video


class exercise_3_workflow():
    """ A workflow of the second exercise. The purpose of this exercise is to implement a basic simulated annealing algorithm to minimize a function.

        Args:
            rng (int): random seed (or random generator).
    """

    def __init__(self, rng=np.random.default_rng(420)):
        self.rng = rng
        self.grid = None
    
    def simulated_annealing(self, f, N:int,T_min:float,x0,T0:float, min:int=0, max:int=np.inf):
        """This function implements the simulated annealing algorithm to find the global minimum
            of a function on an input set included in N and finite.

        Args:
            f (function): function to be minimized (f:N->R).
            N (int): max number of iterations.
            T_min (float): maximal precision
            x0 (_type_): starting point.
            T0 (float): starting temperature.
            min (int, optional): minimum point of the entry set of f. Defaults to 0.
            max (int, optional): maximum point of the entry set of f. Defaults to np.inf.
        """
        n = 0
        xn = x0
        Tn=T0
        X,F=[],[]

        while n<N and Tn>T_min:
            u = round(self.rng.uniform())
            if xn==0:
                y = xn + 1
            elif xn==max:
                    y = xn - 1
            else:
                y = xn + 2*u - 1
            
            if f(y)<f(xn) or (f(y)>=f(xn) and self.rng.uniform() < np.exp(-(f(y)-f(xn))/Tn)):
                xn = y
            X.append(xn)
            F.append(f(xn))
            Tn = T0/np.log(n+2)
            n=n+1
        self.X = X
        self.F = F
    
    def plot_annealing(self):
        """
        This function displays the evolution of the point value during the iterations.
        """
        plt.plot(self.X)
        plt.title("Research of the minimum of f function")
        plt.xlabel("Step")
        plt.ylabel("x")
        plt.show()

    
    def init_city(self, K:int,max_size_city:int,std:float):
        """
        This function generate the city for the traveling problem. The coordinates of the cities are stored in the variable grid.

        Args:
            K (int): Number of cities in the map
            max_size_city (int): Max size of the map (square)
            std (float): Variance used to generate the cities.
        """
        self.K = K
        self.max_city_size = max_size_city
        self.grid = self.rng.integers(max_size_city, size=(K, 2)) + self.rng.normal(scale = std,size = (K,2))

    def distance(self, sigma,dist):
        """
        This function computes the sum of the distances between two consecutive points of the perutation.

        Args:
            sigma (array): a permutation of the cities
            dist (function): function used to calculate a distance between two cities

        Returns:
            float: Returns the sum of the distances between two consecutive points of the perutation.
        """
        K=len(sigma)
        return np.sum([dist(self.grid[sigma[j]], self.grid[sigma[(j + 1) % K]]) for j in range(len(self.grid))])

    def generate_indices(self, K:int):
        """This function generate randomly two differents indices between 0 and K-1

        Args:
            K (int): Number of cities.

        Returns:
            tuple(int, int): A tuple where the first value is the smallest index.
        """
        i = self.rng.integers(K, size=1)[0]
        k = i
        while k == i:
            k = self.rng.integers(K, size=1)[0]
        return min(i,k), max(i,k)

    def generate_neighbour(self, sigma):
        """
        This function generates a neighboring permutation from a given permutation.

        Args:
            sigma (array): A permutation of the cities.

        Returns:
            array: A neighboring permutation.
        """
        sigman_nei = np.copy(sigma)
        i,k = self.generate_indices(len(sigma))
        sigman_nei[i:k+1] = np.flip(sigman_nei[i:k+1])
        return sigman_nei


    def plotCity(self):
        """
        This function displays the map if cities have been generated.

        Raises:
            TypeError: error triggered if no cities have been generated.
        """
        plt.grid()
        plt.style.use("ggplot")
        plt.scatter(self.grid[:,0],self.grid[:,1],marker = "o",color = "blue")
        plt.title("Map")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


    def simulated_annealing_traveling(self, f, N:int,T_min:float,x0,T0:float):
        """This function implements the simulated annealing algorithm for a traveling salesman problem.

        Args:
            f (function): function to be minimized.
            N (int): max number of iterations.
            T_min (float): maximal precision
            x0 (_type_): starting point.
            T0 (float): starting temperature.
        """
        n = 0
        xn = x0
        Tn=T0
        X,F=[],[]
        self.f = f
        while n<N and Tn>T_min:
            y = self.generate_neighbour(xn)
            if f(y)<f(xn) or self.rng.uniform() <= np.exp(-(f(y)-f(xn))/Tn):
                xn = y
            X.append(xn)
            F.append(f(xn))
            Tn = T0/np.log(n+2)
            n=n+1
        self.X = X
        self.F = F

    def plot_distance(self):
        """
        This function displays the evolution of the path distance during the simulated annealing algorithm using the generated traveling problem.
        """
        plt.plot(list(map(self.f, self.X)))
        plt.title("Evolution of the path distance during the optimization algorithm")
        plt.xlabel("Step")
        plt.ylabel("Distance")
        plt.show()


    def plot_simulation_traveling(self, interval:int=20):
        """
        This function plot both cities and paths for the simulated annealing algorithm using the generated traveling problem.

        Args:
            interval (int, optional): Interval between two screenshot of the simulation.. Defaults to 20.

        Returns:
            HTML : A video of the simulation.
        """
        
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        
        # set limit for x and y axis
        axes.set_ylim(-2,  self.max_city_size)
        axes.set_xlim(-2, self.max_city_size)
        axes.set_title("Evolution of the optimum path between cities")
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        plt.style.use("ggplot")
        

        def animate(i):
            X = self.X[i*interval]
            axes.cla()
            plt.grid(False)
            axes.set_title("Evolution of the optimum path between cities")
            axes.set_xlabel('x')
            axes.set_ylabel('y')
            for i in range(len(X)-1):
                axes.plot(np.array([self.grid[X[i], 0], self.grid[X[i+1], 0]]), np.array([self.grid[X[i], 1], self.grid[X[i+1], 1]]), '-ok', mfc='C1', mec='C1')
            axes.plot(np.array([self.grid[X[-1], 0], self.grid[X[0], 0]]), np.array([self.grid[X[-1], 1], self.grid[X[0], 1]]), '-ok', mfc='C1', mec='C1')
        
        anim = animation.FuncAnimation(fig, animate,frames=len(self.X)//interval)
        video = HTML(anim.to_html5_video())
        return video

            
            