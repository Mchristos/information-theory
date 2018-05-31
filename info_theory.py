import numpy as np 
from numpy import log2

def H(p_x):
    """
    Compute the entropy of a random variable distributed ~ p(x)
    """
    return -np.sum(p_x*log2(p_x))

def H_cond(P_yx, p_x):
    """ 
    Compute the conditional entropy H(Y|X) given distributions p(y|x) and p(x)
    """
    return -np.sum((P_yx*log2(P_yx))@p_x)

def KL_div(p_x, q_x):
    """
    Compute the KL-divergence between two random variables
    D(p||q) = E_p[log(p(X)/q(X))]

    p_x : defines p(x), shape (dim_X, )  
    q_x : defines q(x), shape (dim_X, )
    """
    if p_x.shape != q_x.shape:
        raise ValueError("p_x and q_x should have the same length")
    return np.sum(p_x * log2(p_x/q_x))

def I(Pxy):
    """
    Compute the mutual information between two random variables related by 
    their joint distribution p(x,y)

    Pxy : defines the joint distribution p(x,y), shape (dim_Y, dim_X)
    """
    p_x = np.sum(Pxy, axis = 0)
    p_y = np.sum(Pxy, axis = 1)
    return KL_div(Pxy, product(p_x, p_y))

def I2(P_yx, p_x):
    """
    Compute the mutual information between two random variables given the conditional distribution p(y|x) and p(x)
    
    P_yx : matrix defining p(y|x), shape (dim_Y, dim_X)  
    p_x :  defines distribution p(x), shape (dim_X,) 
    """
    p_y = (P_yx@p_x).reshape(-1,1)
    Pxy = P_yx/p_y
    return np.sum( (P_yx*log2(Pxy)) @ p_x )

def product(p_x, p_y):
    """
    Compute the product distribution p(x,y) = p(x)p(y) given distributions 
    p(x) and p(y)
    """
    p_x = p_x.reshape(1,-1)
    p_y = p_y.reshape(-1,1)
    Pxy = p_y@p_x
    assert Pxy.shape == (p_y.shape[0], p_x.shape[1]) 
    return Pxy
