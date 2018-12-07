"""
All helper functions for this project are implemented here.

Author: Angad Gill

Code here is inspired by the following papers:
1. Zinkevich, M.(2010).Parallelized stochastic gradient descent.
Advances in Neural Information Processing Systems,
Retrieved from http://papers.nips.cc/paper/4006-parallelized-stochastic-gradient-descent

2. Niu, F. (2011). HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.
Advances in Neural Information Processing Systems,
Retrieved from http://arxiv.org/abs/1106.5730
"""

from sys import stdout

import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import SGDRegressor
from joblib import Parallel, delayed
import threading




def split_data(X_train, y_train, n_jobs, split_per_job, overlap=False):
    """
    Split the data across workers. Outputs a nested list of X_train and y_train
     [[X_train for worker 1, y_train for worker 1], [X_train for worker 2, y_train for worker 2],...]

    Parameters
    ----------
    X_train: Input training data. May be split across workers, see split_per_job
    y_train: Target training dat
    n_jobs: Number of workers
    split_per_job: Fraction of input data that each worker should have
    overlap: Bool. Should there be overlap in the data split across workers, i.e. should the function use bootstraping

    Returns
    -------
    data: Outputs a nested list of X_train and y_train
     [[X_train for worker 1, y_train for worker 1], [X_train for worker 2, y_train for worker 2],...]

    """
    if overlap:  # Bootstrap the input data across workers
        data_size = len(X_train)
        # np.random.choice uses replace=False so that one worker gets unique samples
        splits = [np.random.choice(data_size, size=int(split_per_job*data_size), replace=False) for _ in range(n_jobs)]
        data = zip([X_train[split] for split in splits], [y_train[split] for split in splits])
        data = list(data)
    else:
        if split_per_job != 1/n_jobs:  # Data must be split evenly if there is no overlap
            raise Exception("split_per_job must be equal to 1/n_jobs")
        data = zip(np.split(X_train, n_jobs), np.split(y_train, n_jobs))
        data = list(data)
    return data


def sim_parallel_sgd(X_train, y_train, X_test, y_test,
                     n_iter, n_jobs, split_per_job, n_sync=1,
                     overlap=False, verbose=False):
    """
    Simulate parallel execution of SGDRegressor.

    Parameters
    ----------
    X_train: Input training data. May be split across workers, see split_per_job
    y_train: Target training data
    X_test: Input test data. Used by all workers
    y_test: Target test data
    n_iter: Number of iterations for each worker
    n_jobs: Number of simulated workers
    n_sync: Number of times weights should be syncrhonized, including the one at the end
    split_per_job: Fraction of input data that each worker should have
    overlap: Bool. Should there be overlap in the data split across workers, i.e. should the function use bootstraping

    Returns
    -------
    scores: nested list of scores of each machine in each iteration
        Each element contains scores for each machine. The last being the aggregate score
        e.g.: [[machine 1 score in iter 1, machine 2 score in iter 1, ..., aggregate score in iter 1]
               [machine 1 score in iter 2, machine 2 score in iter 2, ..., aggregate score in iter 2]
               ...
               [machine 1 score in iter n, machine 2 score in iter n, ..., aggregate score in iter n]]
    """

    """ Split data """
    data = split_data(X_train, y_train, n_jobs, split_per_job, overlap)

    """ Simulate parallel execution """
    scores = []  # List containing final output
    costs = []
    thetas = []
    sgds = []  # List of SGDRegressor objects for each "worker"
    for n in range(n_jobs):
        # warm_start=True is important for iterative training
        # sgds += [SGDRegressor(n_iter=1, warm_start=True)]
        sgds += [SGDRegressor(max_iter=1, tol=0.001, warm_start=True)]
    #sgds += [SGDRegressor()]  # For calculating aggregate score for each iteration
    sgds += [SGDRegressor()]  # For calculating aggregate score for each iteration

    for i in range(n_iter):  # Execute iterations one-by-one
        if verbose:
            stdout.write("Iteration: " + str(i))
            stdout.write('\r')

        iter_scores = []
        iter_coefs = []
        iter_intercepts = []
        iter_costs = []


        for n, sgd in enumerate(sgds):  # Fit model for each "worker" one-by-by
            if n < n_jobs:
                sgd.partial_fit(data[n][0], data[n][1])  # partial_fit() allows iterative training
                iter_scores += [sgd.score(X_test, y_test)]
                iter_coefs += [sgd.coef_]
                iter_intercepts += [sgd.intercept_]
                iter_costs += [computeCost(X_test,y_test,sgd.coef_)]
            else:
                # Calcuate aggregate score for this iteration
                iter_costs = np.mean(np.array((iter_costs)),axis=0)
                iter_coefs = np.mean(np.array(iter_coefs), axis=0)
                iter_intercepts = np.mean(np.array(iter_intercepts), axis=0)
                sgd.coef_ = iter_coefs
                sgd.intercept_ = iter_intercepts
                #iter_scores += [sgd.score(X_test, y_test)]

        scores += [iter_scores]
        costs += [iter_costs]
        thetas += [sgd.coef_]

        if i % int(n_iter/n_sync) == 0 and i != 0:  # Sync weights every (n_iter/n_sync) iterations
            if verbose:
                print("Synced at iteration:", i)
            for sgd in sgds[:-1]:  # Iterate through all workers except the last (which is used for aggregates)
                sgd.coef_ = iter_coefs
                sgd.intercept_ = iter_intercepts

    return scores, costs, thetas


def plot_scores(scores, agg_only=True):
    """
    Plot scores produced by the parallel SGD function

    Parameters
    ----------
    scores: nested list of scores produced by sim_parallel_sgd_scores
        Each element contains scores for each machine. The last being the aggregate score
        e.g.: [[machine 1 score in iter 1, machine 2 score in iter 1, ..., aggregate score in iter 1]
                ...[..]]
    agg_only: plot only the aggregated scores -- last value in each nested list

    Returns
    -------
    No return
    """
    scores = np.array(scores).T
    if not agg_only:
        for s in scores[:-1]:
            plt.figure(1)
            plt.plot(range(len(s)), s)
    plt.figure(1)
    plt.plot(range(len(scores[-1])), scores[-1], '--')



""" Parallel implementation """


def psgd_method(args):
    """
    SGD method run in parallel using map.

    Parameters
    ----------
    args: tuple (sgd, data), where
        sgd is SGDRegressor object and
        data is a tuple: (X_train, y_train)

    Returns
    -------
    sgd: object returned after executing .fit()

    """
    sgd, data = args
    X_train, y_train = data
    sgd.fit(X_train, y_train)
    return sgd


def psgd_method_1(sgd, X_train, y_train):
    """
    SGD method run in parallel using map.

    Parameters
    ----------
    args: tuple (sgd, data), where
        sgd is SGDRegressor object and
        data is a tuple: (X_train, y_train)

    Returns
    -------
    sgd: object returned after executing .fit()

    """
    sgd.fit(X_train, y_train)
    return sgd


def psgd_method_2(sgd, loop_iter, coef, intercept, X_train, y_train):
    """
    SGD method run in parallel using map.

    Parameters
    ----------
    args: tuple (sgd, data), where
        sgd is SGDRegressor object and
        data is a tuple: (X_train, y_train)

    Returns
    -------
    sgd: object returned after executing .fit()

    """

    for _ in range(loop_iter):
        sgd.coef_ = coef
        sgd.intercept_ = intercept
        sgd.fit(X_train, y_train)
        coef = sgd.coef_
        intercept = sgd.intercept_
    return sgd


def parallel_sgd(pool, sgd, n_iter, n_jobs, n_sync, data):
    """
    High level parallelization of SGDRegressor.

    Parameters
    ----------
    pool: multiprocessor pool to use for this parallelization
    sgd: SGDRegressor instance whose coef and intercept need to be updated
    n_iter: number of iterations per worker
    n_jobs: number of parallel workers
    n_sync: number of synchronization steps. Syncs are spread evenly through out the iterations
    data: list of (X, y) data for the workers. This list should have n_jobs elements

    Returns
    -------
    sgd: SGDRegressor instance with updated coef and intercept
    """
    # eta = sgd.eta0*n_jobs
    eta = sgd.eta0
    n_iter_sync = n_iter/n_sync  # Iterations per model between syncs
    sgds = [SGDRegressor(warm_start=True, n_iter=n_iter_sync, eta0=eta)
            for _ in range(n_jobs)]

    for _ in range(n_sync):
        args = zip(sgds, data)
        sgds = pool.map(psgd_method, args)
        coef = np.array([x.coef_ for x in sgds]).mean(axis=0)
        intercept = np.array([x.intercept_ for x in sgds]).mean(axis=0)
        for s in sgds:
            s.coef_ = coef
            s.intercept_ = intercept


    sgd.coef_ = coef
    sgd.intercept_ = intercept

    return sgd


def psgd_1(sgd, n_iter_per_job, n_jobs, X_train, y_train):
    """
    Parallel SGD implementation using multiprocessing. All workers sync once after running SGD independently for
    n_iter_per_job iterations.

    Parameters
    ----------
    sgd: input SGDRegression() object
    n_iter_per_job: number of iterations per worker
    n_jobs: number of parallel processes to run
    X_train: train input data
    y_train: train target data

    Returns
    -------
    sgd: the input SGDRegressor() object with updated coef_ and intercept_
    """

    sgds = Parallel(n_jobs=n_jobs)(
        delayed(psgd_method_1)(s, X_train, y_train)
        for s in [SGDRegressor(n_iter=n_iter_per_job) for _ in range(n_jobs)])
    sgd.coef_ = np.array([x.coef_ for x in sgds]).mean(axis=0)
    sgd.intercept_ = np.array([x.intercept_ for x in sgds]).mean(axis=0)
    return sgd


def psgd_2(sgd, n_iter_per_job, n_jobs, n_syncs, X_train, y_train):
    """
    Parallel SGD implementation using multiprocessing. All workers sync n_syncs times while running SGD independently
    for n_iter_per_job iterations.

    Parameters
    ----------
    sgd: input SGDRegression() object
    n_iter_per_job: number of iterations per worker
    n_jobs: number of parallel processes to run
    n_syncs: number of syncs
    X_train: train input data
    y_train: train target data

    Returns
    -------
    sgd: the input SGDRegressor() object with updated coef_ and intercept_

    """
    # n_syncs = n_jobs
    n_iter_sync = n_iter_per_job/n_syncs  # Iterations per model between syncs

    sgds = [SGDRegressor(warm_start=True, n_iter=n_iter_sync)
            for _ in range(n_jobs)]

    for _ in range(n_syncs):
        sgds = Parallel(n_jobs=n_jobs)(
            delayed(psgd_method_1)(s, X_train, y_train) for s in sgds)
        coef = np.array([x.coef_ for x in sgds]).mean(axis=0)
        intercept = np.array([x.intercept_ for x in sgds]).mean(axis=0)
        for s in sgds:
            s.coef_ = coef
            s.intercept_ = intercept

    sgd.coef_ = coef
    sgd.intercept_ = intercept

    return sgd


def psgd_(sgd, n_iter_per_job, n_jobs, n_syncs, X_train, y_train):
    """
    Parallel SGD implementation using multiprocessing. All workers sync n_syncs times while running SGD independently
    for n_iter_per_job iterations. Each worker will have an increased learning rate -- multiple of n_jobs.

    Parameters
    ----------
    sgd: input SGDRegression() object
    n_iter_per_job: number of iterations per worker
    n_jobs: number of parallel processes to run
    n_syncs: number of syncs
    X_train: train input data
    y_train: train target data

    Returns
    -------
    sgd: the input SGDRegressor() object with updated coef_ and intercept_

    """
    n_iter_sync = n_iter_per_job/n_syncs  # Iterations per model between syncs
    eta = sgd.eta0 * n_jobs

    sgds = [SGDRegressor(warm_start=True, n_iter=n_iter_sync, eta0=eta)
            for _ in range(n_jobs)]

    for _ in range(n_syncs):
        sgds = Parallel(n_jobs=n_jobs)(
            delayed(psgd_method_1)(s, X_train, y_train) for s in sgds)
        coef = np.array([x.coef_ for x in sgds]).mean(axis=0)
        intercept = np.array([x.intercept_ for x in sgds]).mean(axis=0)
        for s in sgds:
            s.coef_ = coef
            s.intercept_ = intercept

    sgd.coef_ = coef
    sgd.intercept_ = intercept

    return sgd


def psgd_4(sgd, n_iter_per_job, n_jobs, X_train, y_train, coef, intercept):
    """
    Parallel SGD implementation using multithreading. All workers read coef and intercept from share memory,
    process them, and then overwrite them.

    Parameters
    ----------
    sgd: input SGDRegression() object
    n_iter_per_job: number of iterations per worker
    n_jobs: number of parallel processes to run
    X_train: train input data
    y_train: train target data
    coef: randomly initialized coefs stored in shared memory
    intercept: randomly initialized intercept stored in shared memory

    Returns
    -------
    sgd: the input SGDRegressor() object with updated coef_ and intercept_
    """
    sgds = [SGDRegressor(warm_start=True, n_iter=1)
            for _ in range(n_jobs)]

    sgds = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(psgd_method_2) (s, n_iter_per_job, coef, intercept, X_train, y_train)
        for s in sgds)

    sgd.coef_ = np.array([x.coef_ for x in sgds]).mean(axis=0)
    sgd.intercept_ = np.array([x.intercept_ for x in sgds]).mean(axis=0)
    return sgd


def plot_speedup_acc(speedup_acc, score1):
    num_procs = speedup_acc[0].shape[0]

    plt.figure(figsize=(15, 3))

    plt.subplot(1, 3, 1)
    plt.plot(range(1,num_procs+1), speedup_acc[0])
    plt.axhline(1, c='r')
    plt.xlabel("Processors")
    plt.xticks(range(1,num_procs+1))
    plt.ylabel("Speed-up")
    plt.yticks(np.arange(0,num_procs+1, 0.5))

    plt.subplot(1, 3, 2)
    plt.plot(range(1,num_procs+1), speedup_acc[1])
    plt.axhline(y=score1, c='r')
    plt.xlabel("Processors")
    plt.xticks(range(1,num_procs+1))
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0,1.25, 0.25))

    plt.subplot(1, 3, 3)
    plt.scatter(speedup_acc[0], speedup_acc[1])
    plt.xlabel("Speed-up")
    plt.xticks(np.arange(0,num_procs+1, 0.5))
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0,1.25, 0.25))

def computeCost(X, y, theta=[[0],[0]]):
    m = y.size
    J = 0  
    h = X.dot(theta) 
    J = 1/(2*m)*np.sum(np.square(h-y)) 
    return(J)

def plot_contour(X,y,theta,label=0):
        # Create grid coordinates for plotting
    theta_0 = np.linspace(-0.5, 0.5, 50)
    theta_1 = np.linspace(-0.5, 0.5, 50)
    theta_x, theta_y = np.meshgrid(theta_0, theta_0, indexing='xy')
    Z = np.zeros((theta_0.size,theta_1.size))



    # Calculate Z-values (Cost) based on grid of coefficients
    for (i,j),v in np.ndenumerate(Z):
        Z[i,j] = computeCost(X,y, theta=[[theta_x[i,j]], [theta_y[i,j]]])
    

    plt.figure()
    fig, ax = plt.subplots()
    CS = ax.contour(theta_x, theta_y, Z, np.logspace(np.log10(np.amin(Z)-1),np.log10(np.amax(Z)+1),15))
    ax.clabel(CS, inline=1, fontsize=10)
    for t in theta:
        ax.scatter(t[0],t[1], c='r')
    
    ax.set_title('Curvas de nível (label: {})'.format(label))
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
