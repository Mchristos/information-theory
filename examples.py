import numpy as np 
from scipy.stats import norm 
import matplotlib.pyplot as plt
from info_theory import *  

class Examples(object):
    def example1():
        """ Illustrate the KL-divergence between normal distributions """
        x = np.linspace(-5,5,100)
        p_x = norm.pdf(x)
        q_x1 = norm.pdf(x-0.1)
        q_x2 = norm.pdf(x-1)
        q_x3  = norm.pdf(x + 1)
        qs = [q_x1, q_x2, q_x3]
        kl_divs = [KL_div(p_x, q) for q in qs]
        plt.plot(x, p_x)
        plt.plot(x, q_x1, x, q_x2, x, q_x3, linestyle = '--' )
        plt.title("KL divergence")
        plt.legend(['p(x)', *kl_divs])
        plt.show()
    
    def example2():
        """Computing mutual information associated with a joint distribution"""
        eps = 1e-12
        # define joint distribution p(x,y)
        Pxy = np.array([
            [1/8,  1/16, 1/32, 1/32],
            [1/16, 1/8,  1/32, 1/32],
            [1/16, 1/16, 1/16, 1/16],
            [1/4,  eps,    eps,    eps ]
        ])
        print("joint distribution in form (y, x)")
        print(Pxy)
        # apriori distribution p(x) 
        p_x = np.sum(Pxy, axis = 0)
        # conditional distribution p(y|x)
        P_yx = Pxy / p_x
        print("mutual information using joint distribution: ")
        print("I = %r" % I(Pxy))
        print("mutual information using conditional distribution: ")
        print("I = %r" % I2(p_x, P_yx))


if __name__ == '__main__':
    Examples.example2()