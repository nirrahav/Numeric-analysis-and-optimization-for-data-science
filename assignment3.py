"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random


class Assignment3:
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

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001):
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

    def leggauss(self, n):
        """
        Generates the Gaussian quadrature points and weights for the Legendre polynomial of order n.
        :param n: The order of the Legendre polynomial
        :return: A tuple of arrays (points, weights) where points is an array of n Gaussian quadrature points,
                 and weights is an array of n Gaussian quadrature weights.
        """
        # Define a small value for the tolerance of the iterations
        eps = 3e-14

        # Calculate the number of points to use in the Gaussian quadrature
        m = int((n + 1) // 2)

        # Initialize arrays to store the Gaussian quadrature points and weights
        x = np.zeros(n)
        w = np.zeros(n)

        # Iterate over half of the points
        for i in range(m):
            # Calculate an initial guess for the Gaussian quadrature point
            z = np.cos(np.pi * (i + 0.75) / (n + 0.5))

            # Use Newton-Raphson method to find the Gaussian quadrature point
            while True:
                # Calculate the Legendre polynomial using a three-term recurrence relation
                p1 = 1.0
                p2 = 0.0
                for j in range(n):
                    p3 = p2
                    p2 = p1
                    p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1)

                # Calculate the derivative of the Legendre polynomial
                pp = n * (z * p1 - p2) / (z * z - 1.0)

                # Update the guess for the Gaussian quadrature point using Newton-Raphson method
                z1 = z
                z = z1 - p1 / pp

                # Check if the tolerance has been met and break out of the loop
                if abs(z - z1) <= eps:
                    break

            # Store the Gaussian quadrature point and its corresponding weight
            x[i] = -z
            x[n - 1 - i] = z
            w[i] = 2.0 / ((1.0 - z * z) * pp * pp)
            w[n - 1 - i] = w[i]

        # Return the Gaussian quadrature points and weights
        return x, w

    def simpson(self, f, a, b, n=100):
        """
        Uses Simpson's rule to approximate the definite integral of f(x) over the interval [a, b].
        n is the number of subintervals to use, and should be even.
        """

        # If an equals b, then the interval has no width and the integral is zero.
        if a == b:
            return 0

        # If n is odd, then Simpson's rule cannot be applied, and this function raises an error.
        if n % 2 != 0:
            raise ValueError("n must be even")

        # Compute the width of each sub-interval.
        h = (b - a) / n

        # Generate a list of n+1 equally spaced x-values that span the interval [a, b].
        x = [a + i * h for i in range(n + 1)]

        # Evaluate the function f at each x-value to obtain a list of n+1 corresponding function values.
        fx = [f(x[i]) for i in range(n + 1)]

        # Use Simpson's rule to compute the integral of f over the interval [a, b].
        # This formula uses the values of f at the endpoints and at even and odd midpoints of the sub-intervals.
        integral = h / 3 * (fx[0] + fx[n] + 4 * sum(fx[1:n:2]) + 2 * sum(fx[2:n - 1:2]))

        # Return the computed integral.
        return integral


    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the integration error.
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions.

        Integration error will be measured compared to the actual value of the
        definite integral.

        Note: It is forbidden to call f more than n times.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        # replace this line with your solution
        # gaussian_quadrature
        # Define the quadrature points and weights
        if a == b:
            return 0
        x, w = self.leggauss(n)

        # Map the quadrature points from [-1, 1] to [a, b]
        x = 0.5 * (b - a) * x + 0.5 * (b + a)

        # Evaluate the function at the quadrature points and multiply by the weights
        for index in range(len(x)):
            x[index] = f(x[index])
        result = 0.5 * (b - a) * np.sum(w * x)

        return np.float32(result)

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area.

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """
        points = self.intersections(f1, f2, 1, 100)

        # The variable n stores the number of points where f1 and f2 intersect within the defined domain.
        n = len(points)

        # If there are less than 2 points of intersection, it is not possible to compute the area of intersection,
        # so this function returns NaN (not a number).
        if n < 2:
            return np.NaN

        # Define a new function func that is the difference between f1 and f2. This will be used to compute the area
        # of the intersection by integrating the absolute value of func over the x-axis within the intersection domain.
        func = lambda x: f1(x) - f2(x)

        # Initialize the result variable to 0. This variable will store the accumulated area of the intersection.
        result = 0

        # Iterate over the intersection points, and compute the area between adjacent points using Simpson's rule.
        for index in range(len(points) - 1):

            # If the distance between two adjacent points is less than 0.001, then they are considered to be too
            # close to each other and do not contribute to the intersection area. So, skip over this pair of points.
            if abs(points[index] - points[index + 1]) < 0.001:
                continue

            # If the distance between two adjacent points is greater than or equal to 0.001, then compute the area
            # between these two points using Simpson's rule, and add it to the result variable.
            else:
                result += abs(self.simpson(func, points[index], points[index + 1]))

        # Convert the result to a 32-bit floating point number, and return it.
        return np.float32(result)



##########################################################################

import math
import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):
    def test_Area_between(self):
        ass3 = Assignment3()
        f1 = lambda x: (x - 10) ** 2
        f2 = lambda x: 4
        t1 = time.time()
        S = ass3.areabetween(f1, f2)
        t2 = time.time()
        print('area =', S)
        print((t2 - t1))

        f1 = lambda x: math.sin(math.log(x))
        f2 = lambda x: pow(x, 2) - 3 * x + 2
        t1 = time.time()
        S = ass3.areabetween(f1, f2)
        t2 = time.time()
        print('area =', S)
        print((t2 - t1))
        print('Error =', S - 0.731257)
    # def test_integrate_float32(self):
    #     ass3 = Assignment3()
    #     f1 = np.poly1d([-1, 0, 1])
    #     r = ass3.integrate(f1, -1, 1, 10)
    #
    #     self.assertEquals(r.dtype, np.float32)


#    def test_integrate_hard_case(self):
#        ass3 = Assignment3()
#        f1 = strong_oscilations()
#        r = ass3.integrate(f1, 0.09, 10, 20)
#        true_result = -7.78662 * 10 ** 33
#        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
