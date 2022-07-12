import numpy as np
from scipy.interpolate import CubicSpline
from sympy import ShapeError
from Lan_sims.tools import *
from tools import Runge_Kutta_integrator_GLE, Runge_Kutta_integrator_LE


class Lan_integrator():
    """
    Main class for Langevin simulations.
    """

    def __init__(self, 
        free_energy, 
        gammas, 
        couplings=None, 
        dt=0.01, 
        segment_length=int(1e7), 
        number_segments=1,
        m=1, 
        stride=1,
        single_particle=True,
        kT=2.494
        ):
        self.dt = dt
        self.segment_len = segment_length
        self.number_segments = number_segments
        self.m = m
        self.stride = stride
        self.single_particle = single_particle
        if free_energy.shape[1] != 2:
            raise ShapeError('free energy array must contain two columns (positions, energies)!')
        else:
            self.fe = free_energy[:,1]
            self.edges = free_energy[:,0]
        self.gammas = gammas
        self.couplings = couplings
        self.initials = None
        self.amat = None
        self.bin_width = None

    def parse_input(self):
        if not isinstance(self.gammas, np.ndarray):
            if isinstance(self.gammas, (int, float)):
                self.gammas = np.array([self.gammas])
            elif isinstance(self.gammas, list):
                self.gammas = np.array(self.gammas)
            else:
                raise TypeError('Give friction coefficients as scalar, list of scalars or numpy.ndarray!')
        if len(self.gammas) > 1 and self.couplings == None:
            raise TypeError('Coupling coefficients needed!')
        elif not isinstance(self.couplings, np.ndarray):
            if isinstance(self.couplings, (int, float)):
                self.couplings = np.array([self.couplings])
            elif isinstance(self.gammas, list):
                self.couplings = np.array(self.couplings)
            else:
                raise TypeError('Give friction coefficients as scalar, list of scalars or numpy.ndarray!')
        if len(self.gammas) != len(self.couplings) + 1:
            raise ShapeError('length friction coeffs must equal to length coupling coeffs + 1')
        else:
            self.gammas = np.append(self.gammas[::-1], np.zeros(1))[::-1]


    def gen_initial_values(self):
        self.initials = np.random.randn(len(self.gammas) + 1)

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
        self.parse_input()
        self.gen_initial_values()
        self.spline_free_energy()
        if self.single_particle:
            for segment in range(self.number_segments):
                x, self.initials = Runge_Kutta_integrator_LE(
                    self.segment_len, 
                    self.dt, 
                    self.m, 
                    self.gammas, 
                    self.initials, 
                    self.edges, 
                    self.amat)
        else:
            for segment in range(self.number_segments):
                x, self.initials = Runge_Kutta_integrator_GLE(
                    self.segment_len, 
                    self.dt, 
                    self.m, 
                    self.gammas, 
                    self.couplings,
                    self.initials, 
                    self.edges, 
                    self.amat)