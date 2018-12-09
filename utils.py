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

import pymp

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

def sim_parallel_sgd_p(X_train, y_train, X_test, y_test,
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
    #real_iter = int(n_iter/split_per_job)

    """ Simulate parallel execution """
    scores = []  # List containing final output
    costs = []
    thetas = []
    sgds = []  # List of SGDRegressor objects for each "worker"
    n_feature = X_train.shape[1]

    for n in range(n_jobs):
        # warm_start=True is important for iterative training
        # sgds += [SGDRegressor(n_iter=1, warm_start=True)]
        sgds += [SGDRegressor(max_iter=1, tol=0.001, warm_start=True)]
    #sgds += [SGDRegressor()]  # For calculating aggregate score for each iteration
    sgds += [SGDRegressor()]  # For calculating aggregate score for each iteration

    print(n_iter)
    for i in range(n_iter):  # Execute iterations one-by-one
        
        if verbose:
            stdout.write("Iteration: " + str(i))
            stdout.write('\r')

        iter_scores  = pymp.shared.array((n_jobs,))
        iter_costs = pymp.shared.array((n_jobs,))
        iter_intercepts = pymp.shared.array((n_jobs,n_feature))
        iter_coefs = pymp.shared.array((n_jobs,n_feature))


        with pymp.Parallel(n_jobs) as p:
            for n in p.range(n_jobs):
                sgd = sgds[n]
                if n < n_jobs:
                    sgd.partial_fit(data[n][0], data[n][1])  # partial_fit() allows iterative training
                    iter_scores[n] = sgd.score(X_test, y_test)
                    iter_coefs[n] = sgd.coef_
                    iter_intercepts[n] = sgd.intercept_
                    iter_costs[n] = computeCost(X_test,y_test,sgd.coef_)


        # Calcuate aggregate score for this iteration
        #print("Iter:{}|cost:{}".format(i,iter_costs))

        iter_costs = np.mean(np.array((iter_costs)),axis=0)
        iter_coefs = np.mean(np.array(iter_coefs), axis=0)
        iter_intercepts = np.mean(np.array(iter_intercepts), axis=0)
        sgds[n].coef_ = iter_coefs
        sgds[n].intercept_ = iter_intercepts
        #iter_scores += [sgd.score(X_test, y_test)]
    #p.print(p.thread_num)

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

def sim_parallel_sgd_c4(X_train, y_train, X_test, y_test,
                     n_iter, n_jobs, split_per_job, n_sync=1,
                     overlap=False, verbose=False, sync_type=0):
    """
    paraleliza todo data set
    parallel execution of SGDRegressor. 

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
    sync_type: if 0 sync by the mean, if 1 sync by min cost

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
    n_perSync = int(n_iter/n_sync)

    """Check if compatible"""
    if (n_perSync*n_sync != n_iter):
        print("n_perSync:{}|n_sync:{}|n_iter:{}".format(n_perSync,n_sync,n_iter))
        raise Exception("Number of syncs has to be multiple of number of iter")

    """ Simulate parallel execution """
    scores = pymp.shared.array((n_iter,n_jobs))  # List containing final output
    costs = pymp.shared.array((n_iter,n_jobs))
    #thetas = pymp.shared.array((n_iter,X_train.shape[1]))
    sgds = []  # List of SGDRegressor objects for each "worker" (firstPrivet)
    for n in range(n_jobs):
        sgds += [SGDRegressor(max_iter=1, tol=0.001, warm_start=True)]
    
    n_features = len(data[n][0][0])

    for s in range(n_sync):
        if verbose:
            stdout.write(f"sync: {s} de {n_sync}")
            stdout.write('\r')
        coefs = pymp.shared.array((n_jobs,n_features))
        intercepts = pymp.shared.array((n_jobs,1))

        with pymp.Parallel(n_jobs) as p:
            for i in range(n_perSync):
                sgds[p.thread_num].partial_fit(data[n][0], data[n][1])  # partial_fit() allows iterative training
                scores[n_perSync*s+i][p.thread_num] = sgds[p.thread_num].score(X_test, y_test) 
                costs[n_perSync*s+i][p.thread_num] = computeCost(X_test,y_test,sgds[p.thread_num].coef_)
                if i == n_perSync-1:
                    #print(coefs[p.thread_num])
                    coefs[p.thread_num] = sgds[p.thread_num].coef_
                    intercepts[p.thread_num] = sgds[p.thread_num].intercept_

        #sincronia pela media     
        if sync_type == 0:
            iter_coefs = np.mean(coefs,axis=0)
            iter_intercepts = np.mean(intercepts,axis=0)
        if sync_type == 1:
            index = np.argmin(costs[n_perSync*(s+1)-1])
            iter_coefs = coefs[index]
            iter_intercepts = intercepts[index]

        #sincronia pelo minimo

        if verbose:
            print("Synced at iteration:", n_perSync*s+i+1)
        for sgd in sgds:  # Iterate through all workers except the last (which is used for aggregates)
            sgd.coef_ = iter_coefs
            sgd.intercept_ = iter_intercepts

    costs = np.mean(costs,axis=1)
    return scores, costs, 1


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

def computeCost(X, y, theta=[[0],[0]]):
    m = y.size
    J = 0  
    h = X.dot(theta) 
    J = 1/(2*m)*np.sum(np.square(h-y)) 
    return(J)

def plot_contour(X,y,theta,label=0):
        # Create grid coordinates for plotting
    theta_0 = np.linspace(-20, 20, 50)
    theta_1 = np.linspace(-20, 20, 50)
    theta_x, theta_y = np.meshgrid(theta_0, theta_0, indexing='xy')
    Z = np.zeros((theta_0.size,theta_1.size))



    # Calculate Z-values (Cost) based on grid of coefficients
    for (i,j),v in np.ndenumerate(Z):
        Z[i,j] = computeCost(X,y, theta=[[theta_x[i,j]], [theta_y[i,j]]])
    

    plt.figure()
    fig, ax = plt.subplots()
    CS = ax.contour(theta_x, theta_y, Z, np.logspace(np.log10(np.amin(Z)-1),np.log10(np.amax(Z)+1),15))
    ax.clabel(CS, inline=1, fontsize=10)
    ax.scatter(theta[0],theta[1], c='r')
    ax.set_title('Curvas de nível (label: {})'.format(label))
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
