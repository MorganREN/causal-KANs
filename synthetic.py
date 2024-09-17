import numpy as np
from scipy.integrate import odeint
import math
import pandas as pd

def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta


def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.01, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += + errors[:, t-1]

    return X.T[burn_in:], beta, GC


def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.01, burn_in=1000,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC



# Zexuan's method

def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
def generate_ds1(total_length, variance=0.01): 
    np.random.seed(0)
    z = [4,1,2,1]
    x=[2,4,1,3]
    #y=[3,3,3,3]
    p = []
    n = [3,3,3,3]
    y_gc = [3,3,3,3]
    y_ngc = [3,3,3,3]
    for i in range(4,total_length+4):
        z.append(math.tanh(z[i-1]+np.random.normal(0,variance)))
        p.append(z[i]**2+np.random.normal(0,0.05))
        x.append(math.sin(x[i-1])+np.random.normal(0,variance))
    
        term1 = sigmoid(z[i-4])
        term2 = sigmoid(x[i-4])
        # term1 = z[i-4]*z[i-3]
        # term2 = x[i-2]*x[i-1] 
        y_gc.append(term1 + term2 + np.random.normal(0,variance))
        y_ngc.append(term1 + np.random.normal(0,variance))
        n.append(np.random.normal(0,1))
        #noise = np.random.normal(0,1)
        #alpha = (abs(term1+term2)/sig_to_noise)/abs(noise)
        #y.append(term1+term2+alpha*noise)

    x=x[-total_length:]
    p=p[-total_length:]
    z=z[-total_length:]
    y_gc=y_gc[-total_length:]
    y_ngc=y_ngc[-total_length:]
    # df=pd.DataFrame({"x":x,"y":y,"p":p,"z":z})
    df = pd.DataFrame({'z': z, 'u': p, 'x': x, 'y_ngc': y_ngc, 'y_gc': y_gc})
    return df