import numpy as np
from scipy.interpolate import CubicSpline
from sympy import ShapeError
from Lan_sims.tools import *


class Lan_integrator():
    """
    Main class for Langevin simulations.
    """

    def __init__(self, free_energy, stride=1, dt=0.01, m=1, gamma=1, kT=2.494):
        self.dt = dt
        self.stride = stride
        if free_energy.shape[1] != 2:
            raise ShapeError('free energy array must contain two columns (positions, energies)!')
        else:
            self.fe = free_energy[:,1]
            self.edges = free_energy[:,0]
        self.x0 = None
        self.v0 = None
        self.amat = None
        self.bin_width = None


    def gen_initial_values(self):
        self.x0 = np.random.randn()

    def cubic_polynom(self, x, index):
        output = sum(
            self.amat[index, order] * (x - self.edges[index]) ** (3 - order) 
            for order in range(4))
        return output

    def cubic_polynom_der(self, x, index):
        output = sum(
            (3 - order) * self.amat[index, order] * (x - self.edges[index]) ** (2 - order) 
            for order in range(3))
        return output


    def spline_free_energy(self):
        cs = CubicSpline(self.edges[::self.stride], self.fe[::self.stride], bc_type='not-a-knot')
        self.amat = cs.c.T
        

    def integrate_LE(self):
        self.x = 0