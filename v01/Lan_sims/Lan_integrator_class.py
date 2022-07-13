from typing import Type
import numpy as np
from scipy.interpolate import CubicSpline
from sympy import ShapeError
from Lan_sims.tools import *


class Lan_integrator():
    """
    Main class for Langevin simulations.
    """

    def __init__(self, 
        free_energy, 
        gammas, 
        couplings = None, 
        dt = 0.01, 
        segment_length = int(1e7), 
        number_segments = 1,
        m = 1., 
        stride = 1,
        nbins = 80,
        hist_range = None,
        single_particle = True,
        kT = 2.494
        ):
        self.dt = dt
        self.segment_len = segment_length
        self.number_segments = number_segments
        self.m = m
        self.stride = stride
        self.nbins = nbins
        self.single_particle = single_particle
        if free_energy.shape[1] != 2:
            raise ShapeError('free energy array must contain two columns (positions, energies)!')
        else:
            self.fe = free_energy[:,1]
            self.edges = free_energy[:,0]
        if not self.single_particle and self.segment_len > 1:
            if hist_range != None:
                self.hist_range = hist_range
            else:
                raise TypeError('hist_range is required for continuation!')
        self.gammas = gammas
        self.couplings = couplings
        self.initials = None
        self.amat = None
        self.bin_width = None
        self.kT=kT
        self.x = None
        self.histogram = np.zeros(self.nbins)

    def parse_input(self):
        if not self.single_particle:

            if not isinstance(self.gammas, np.ndarray):
                if isinstance(self.gammas, (int, float)):
                    self.gammas = np.array([self.gammas])
                elif isinstance(self.gammas, list):
                    self.gammas = np.array(self.gammas)
                else:
                    raise TypeError('Give friction coefficients as scalar, list of scalars or numpy.ndarray!')
        
            if self.couplings == None:
                raise TypeError('Coupling coefficients needed!')
            elif not isinstance(self.couplings, np.ndarray):
                if isinstance(self.couplings, (int, float)):
                    self.couplings = np.array([self.couplings])
                elif isinstance(self.gammas, list):
                    self.couplings = np.array(self.couplings)
                else:
                    raise TypeError('Give coupling coefficients as scalar, list of scalars or numpy.ndarray!')

            if len(self.gammas) != len(self.couplings) + 1:
                raise ShapeError('length friction coeffs must equal to length coupling coeffs + 1')
    
            self.gammas = np.append(self.gammas[::-1], np.zeros(1))[::-1]


            if isinstance(self.hist_range, (tuple, list)):
                if isinstance(self.hist_range, list):
                    self.hist_range = tuple(self.hist_range)
            else:
                raise TypeError('hist_range must be list or tuple!')

    def gen_initial_values(self):
        if self.single_particle:
            self.initials = np.zeros(2)
        else: 
            self.initials = np.zeros(len(self.gammas))
        x0 = self.edges[self.fe == np.min(self.fe)]
        v0 = np.random.normal(0., np.sqrt(self.kT / self.m))
        self.initials[0] = x0
        self.initials[1] = v0
        if not self.single_particle:
            for index, coupling in enumerate(self.couplings):
                self.initials[2+index] = np.random.normal(x0, np.sqrt(self.kT / coupling))


    def spline_free_energy(self):
        cs = CubicSpline(self.edges[::self.stride], self.fe[::self.stride], bc_type='not-a-knot')
        self.amat = cs.c.T

    def compute_distribution(self):
        if self.single_particle:
            self.histogram, self.sim_edges = np.histogram(self.x, bins=self.nbins)
        else:
            hist_dummy, self.sim_edges = np.histogram(self.x, bins=self.nbins, range=self.hist_range)
            self.histogram += hist_dummy

    def compute_free_energy(self):
        self.fe_pos =(self.sim_edges[1:] + self.sim_edges[:]) / 2
        self.fe_sim = -self.kT * np.log(self.histogram)
        self.fe_sim -= np.min(self.fe_sim)
        

    def integrate_LE(self):
        self.parse_input()
        self.gen_initial_values()
        self.spline_free_energy()
        if self.single_particle:
            for segment in range(self.number_segments):
                self.x, self.initials = Runge_Kutta_integrator_LE(
                    self.segment_len, 
                    self.dt, 
                    self.m, 
                    self.gammas, 
                    self.initials, 
                    self.edges, 
                    self.amat)
                #self.compute_distribution()
        else:
            for segment in range(self.number_segments):
                self.x, self.initials = Runge_Kutta_integrator_GLE(
                    self.segment_len, 
                    self.dt, 
                    self.m, 
                    self.gammas, 
                    self.couplings,
                    self.initials, 
                    self.edges, 
                    self.amat)
                #self.compute_distribution()
        #self.compute_free_energy()