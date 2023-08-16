from sampleFunctions import bezier3

"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
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

    def get_bezier_coef(self, points):
        # since the formulas work given that we have n+1 points
        # then n must be this:
        n = len(points) - 1

        # build coefficents matrix
        C = 4 * np.identity(n)
        np.fill_diagonal(C[1:], 1)
        np.fill_diagonal(C[:, 1:], 1)
        C[0, 0] = 2
        C[n - 1, n - 1] = 7
        C[n - 1, n - 2] = 2

        # build points vector
        P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
        P[0] = points[0] + 2 * points[1]
        P[n - 1] = 8 * points[n - 1] + points[n]

        return C, P

    def solve_system(self, points):
        # solve system, find a & b
        n = len(points) - 1
        C, P = self.get_bezier_coef(points)
        A = self.thomas(C, P)
        B = [0] * n
        for i in range(n - 1):
            B[i] = 2 * points[i + 1] - A[i + 1]
        B[n - 1] = (A[n - 1] + points[n]) / 2

        return A, B

    def get_cubic(self, a, b, c, d):
        return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t,
                                                                                                          2) * c + np.power(
            t, 3) * d





    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        # This code calculates the value of a given function f at n evenly spaced points within the interval [a, b].

        if n == 1:
            # If n is equal to 1, return the value of the function at the midpoint of the interval [a,
            # b]. The midpoint is found by evaluating the function at the second point of the linspace (n + 2 points
            # in total).
            return lambda x: f(np.linspace(a, b, n + 2))[1]

        # Create an array of n evenly spaced points within the interval [a, b].
        values_x = np.linspace(a, b, n)

        # Initialize an empty list to store the (x, y) pairs of the function evaluation at each point.
        all_points = []

        # Loop over the values of x.
        for i in range(len(values_x)):
            # Append the (x, y) pair to the list of points.
            all_points.append(np.array([values_x[i], f(values_x[i])]))

        # finding a and b
        A,B = self.solve_system(all_points)


        def find_y(x):
            # loop over the intervals between x values
            for i in range(n - 1):
                # check if x lies in the current interval

                if values_x[i] <= x <= values_x[i + 1]:
                    # normalize x within the interval
                    normal_x = (x - values_x[i]) / (values_x[i + 1] - values_x[i])

                    # Define the cubic function for the interval
                    cubic = self.get_cubic(all_points[i],A[i], B[i], all_points[i + 1])

                    # return the y-coordinate of the interpolated point
                    return cubic(normal_x)[1]

        return find_y


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
