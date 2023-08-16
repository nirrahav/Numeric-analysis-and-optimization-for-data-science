"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random




class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def thomas(self, C, P):
        n = len(P)
        a = [0] + [C[i][i - 1] for i in range(1, n)]
        b = [C[i][i] for i in range(n)]
        c = [C[i][i + 1] for i in range(n - 1)] + [0]
        d = [P[i] for i in range(n)]

        # Forward substitution
        for i in range(1, n):
            b[i] -= a[i] * c[i - 1] / b[i - 1]
            d[i] -= a[i] * d[i - 1] / b[i - 1]

        # Backward substitution
        x = [0] * n
        x[-1] = d[-1] / b[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

        return x


    def calculate_coefficients(self, x, y, d):
        """
        Calculates the coefficients of the polynomial.

        Parameters
        ----------
        x : numpy.ndarray
            Array of time points.
        y : numpy.ndarray
            Array of function values.
        d : int
            Expected degree of the polynomial matching f.

        Returns
        -------
        solution : numpy.ndarray
            Array of polynomial coefficients.
        """
        exponents_sum = np.zeros((d + 1, d + 1))
        for i in range(d + 1):
            for j in range(d + 1):
                exponents_sum[i, j] = np.sum(np.power(x, i + j))
        xy = np.zeros((d + 1, 1))
        for i in range(d + 1):
            xy[i, 0] = np.sum(y * np.power(x, i))
        solution = self.thomas(exponents_sum, xy)

        return solution

    def interpolate_polynomial(self, x, solution):
        """
        Defines a function that represents the interpolated polynomial.

        Parameters
        ----------
        x : float
            Point at which to evaluate the polynomial.
        solution : numpy.ndarray
            Array of polynomial coefficients.

        Returns
        -------
        result : float
            Value of the interpolated polynomial at x.
        """
        result = lambda x, solution: sum(
            [solution[i][0] * np.power(x, i) if i > 0 else solution[i][0] for i in range(len(solution))])
        return result(x, solution)

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            A function which returns an approximate (noisy) Y value given X.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        # Generates time points for the interpolation.
        x = np.linspace(a, b, 4 * maxtime)
        # Calculates the function values at each time point.

        y = np.array([f(d) for d in x])
        solution = self.calculate_coefficients(x, y, d)

        func = lambda x: self.interpolate_polynomial(x, solution)

        return func


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):
            self.assertNotEquals(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)






if __name__ == "__main__":
    unittest.main()
