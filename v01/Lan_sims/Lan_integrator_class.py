import numpy as np
from scipy.interpolate import CubicSpline
from sympy import ShapeError
from Lan_sims.tools import *


class Lan_integrator():
    """
    Main class for Langevin simulations.
    """

    def __init__(self, dt=0.01, m=1, gamma=1, free_energy=None, kT=2.494):
        self.dt = dt
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

    def spline_free_energy(self):
        cs=CubicSpline(self.fe[::2,0], self.fe[::2,1])
        self.amat=cs.c.T
        self.edges=self.fe[::2,0]

    def integrate_LE(self):
        self.x = 0