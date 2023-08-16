"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable
import sympy


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """
        pass

    def get_derivative(self, f):
        # Define a new function that computes the derivative of f at any point x
        def df(x, h=0.0001):
            # Compute the derivative of f at x using the central difference method
            return (f(x + h) - f(x - h)) / (2 * h)

        # Return the new function df that computes the derivative of f
        return df

    def find_root(self, f, a, b, maxerr, max_iter=50):
        """
        This function finds a root of a continuous function f using the combined secant and bisection method.
        :param f: The function whose root we want to find
        :param a: The starting interval [a, b] that contains the root
        :param b: The ending interval [a, b] that contains the root
        :param maxerr: The tolerance for the stopping criteria
        :param max_iter: The maximum number of iterations
        :return: The root of the function
        """
        # Initialize the iteration counter
        k = 0

        while k < max_iter:
            # Evaluate the function at the endpoints of the interval
            fa = f(a)
            fb = f(b)

            # Compute the secant estimate
            xk = b - fb * (b - a) / (fb - fa)

            # Evaluate the function at the secant estimate
            fxk = f(xk)

            # Check if the absolute value of the function at the secant estimate is less than the tolerance
            if np.abs(fxk) < maxerr:
                return xk

            # Check if the sign of the function at the endpoints of the interval has changed
            if fa * fxk < 0:
                b = xk
            elif fb * fxk < 0:
                a = xk
            else:
                # If the sign of the function at the endpoints of the interval has not changed, use the bisection method
                c = (a + b) / 2
                fc = f(c)

                if fc * fa < 0:
                    b = c
                elif fc * fb < 0:
                    a = c

            # Increment the iteration counter
            k += 1

        # If the maximum number of iterations has been reached, return None
        return None

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """
        result = []  # Initialize an empty list to store the roots
        test = np.linspace(a, b, 80)  # Generate 40 evenly spaced test points between a and b
        func = lambda x: f1(x) - f2(x)  # Define the function whose roots we want to find
        dfunc = self.get_derivative(func)  # Compute the derivative of the function
        # Loop over adjacent pairs of test points and find any roots of the function within each interval
        for indexf in range(len(test) - 1):
            rootf = self.find_root(func, test[indexf], test[indexf + 1], maxerr)
            if rootf is None:
                continue  # If no root is found, move on to the next interval
            else:
                result.append(rootf)  # If a root is found, add it to the result list
            rootdf = self.find_root(dfunc, test[indexf], test[indexf + 1], maxerr)
            if rootdf is None:
                continue  # If no root is found, move on to the next interval
            else:
                # Check if the corresponding point on the function is close to zero
                if abs(func(rootdf) - 0) < maxerr:
                    result.append(rootdf)  # If so, add the root to the result list
        return result  # Return the list of roots


##########################################################################


# import unittest
# from sampleFunctions import *
# from tqdm import tqdm
if __name__ == "__main__":
    ass2 = Assignment2()

    f1 = np.poly1d([2, -6, 9])
    f = lambda x: x ** 2 - 6 * x + 9
    df1 = ass2.get_derivative(f)
    try1 = ass2.find_root(df1, 0, 8, 0.001)
    print(abs(0 - f(try1)) < 0.001)
    print(df1)
    f2 = np.poly1d([1, 0, 0])
    t1 = time.time()
    X = ass2.intersections(f1, f2, 0, 10, 0.001)
    print(X)
    t2 = time.time()
    print(t2 - t1)
