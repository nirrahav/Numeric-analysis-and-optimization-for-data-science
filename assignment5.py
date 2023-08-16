"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from functionUtils import AbstractShape
import math
from sklearn.cluster import KMeans
from sampleFunctions import *


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    class MyShape(AbstractShape):
        def __init__(self):
            pass


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """

        n = int(1 / maxerr)  # Calculate the number of points to sample on the contour based on a maximum error allowed.
        points = contour(n)  # Generate n equally-spaced points on the contour.
        x = points[:, 0]  # Extract the x-coordinates of the contour points.
        y = points[:, 1]  # Extract the y-coordinates of the contour points.
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))  # Calculate the area enclosed by
        # the contour using the shoelace formula.
        return np.float32(area)  # Return the area as a single-precision float value.




    def get_ordered_centers(self,centers):
        """
        Sort the given list of centers in clockwise order and return them as a NumPy array.

        Parameters:
        centers: A list of 2-tuples representing the centers to be sorted.

        Returns:
        ndarray: An array of shape (num_centers, 2) representing the sorted centers.
        """
        centers_array = np.array(centers)

        # Calculate the Euclidean distances between each center and the first center using NumPy broadcasting.
        distances = np.sqrt(np.sum((centers_array - centers_array[0]) ** 2, axis=1))

        # Sort the centers based on their distances to the first center.
        indices = np.argsort(distances)
        ordered_centers = centers_array[indices]

        return ordered_centers

    def calculate_polygon_area(self, centers):
        """
        Calculate the signed area of the polygon formed by the given centers.

        Parameters:
        centers: An array of shape (num_centers, 2) representing the centers of the polygon.

        Returns:
        float: The signed area of the polygon.
        """
        x, y = centers.T
        shifted_x, shifted_y = np.roll(x, -1), np.roll(y, -1)
        area = np.abs(np.dot(x, shifted_y) - np.dot(shifted_x, y)) / 2.0
        return area

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        maxtime *= 4500
        maxtime = int(maxtime)
        samp = [list(sample()) for iter in range(maxtime - 1)]
        kmeans = KMeans(n_clusters = 35).fit(samp)
        ordered_centers = self.get_ordered_centers(kmeans.cluster_centers_)
        area = self. calculate_polygon_area(ordered_centers)
        return My_class(area)


class My_class(AbstractShape):

    def __init__(self, area):
        self._area = area

    def area(self):
        return self._area

    def contour(self, n: int):
        w = np.linspace(0, 2 * np.pi, num=n)
        x = np.cos(w) * self._radius + self._cx
        y = np.sin(w) * self._radius + self._cy
        xy = np.stack((x, y), axis=1)
        return xy


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
