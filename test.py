""" unit tests for information theory module """

import unittest
import info_theory as info
import numpy as np

class Test(unittest.TestCase):

    def test_H(self):
        """ correct entropy calculation: uniform distribution"""
        n = 4
        p_x = (1/n)*np.ones(n)
        H = info.H(p_x)
        self.assertEqual(H, np.log2(n))

    def testI(self):
        """ correct mutual information calculation """
        eps = 1e-12
        # define joint distribution p(x,y)
        Pxy = np.array([
            [1/8,  1/16, 1/32, 1/32],
            [1/16, 1/8,  1/32, 1/32],
            [1/16, 1/16, 1/16, 1/16],
            [1/4,  eps,  eps,  eps ]
        ])
        # apriori distribution p(x) 
        p_x = np.sum(Pxy, axis = 0)
        # conditional distribution p(y|x)
        P_yx = Pxy / p_x
        self.assertAlmostEqual(info.I(Pxy), info.I2(P_yx, p_x)) 






if __name__ == "__main__":
    unittest.main()