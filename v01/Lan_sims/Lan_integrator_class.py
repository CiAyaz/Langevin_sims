from cProfile import label
from typing import Type
import numpy as np
from scipy.interpolate import CubicSpline
from sympy import ShapeError
import matplotlib.pyplot as plt
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
        mass = 1., 
        stride = 1,
        nbins = 80,
        hist_range = None,
        single_particle = True,
        kT = 2.494,
        plot=True,
        save=False,
        path_to_save='./',
        single_particle_method="BAOAB"
        ):
        self.dt = dt
        self.segment_len = segment_length
        self.number_segments = number_segments
        self.mass = mass
        self.stride = stride
        self.nbins = nbins
        self.single_particle = single_particle
        if free_energy.shape[1] != 2:
            raise ShapeError('free energy array must contain two columns (positions, energies)!')
        else:
            self.fe = free_energy[:,1]
            self.edges = free_energy[:,0]
        self.hist_range = hist_range
        if self.number_segments > 1:
            if self.hist_range == None:
                raise TypeError('hist_range is required for continuation!')
        self.gammas = gammas
        self.couplings = couplings
        self.initials = None
        self.amat = None
        self.bin_width = None
        self.kT=kT
        self.x = None
        self.histogram = np.zeros(self.nbins)
        self.sim_edges = None
        self.fe_pos = None
        self.fe_sim = None
        self.fe_max = None
        self.plot = plot
        self.save = save
        self.path_to_save = path_to_save
        self.single_particle_method = single_particle_method

    def parse_input(self):
        if not self.single_particle:

            if not isinstance(self.gammas, np.ndarray):
                if isinstance(self.gammas, (int, float)):
                    self.gammas = np.array([self.gammas])
                elif isinstance(self.gammas, list):
                    self.gammas = np.array(self.gammas)
                else:
                    raise TypeError('Give friction coefficients as scalar, list of scalars or numpy.ndarray!')
        
            if not isinstance(self.couplings, np.ndarray):
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
        v0 = np.random.normal(0., np.sqrt(self.kT / self.mass))
        self.initials[0] = x0
        self.initials[1] = v0
        if not self.single_particle:
            for index, coupling in enumerate(self.couplings):
                self.initials[2+index] = np.random.normal(x0, np.sqrt(self.kT / coupling))


    def spline_free_energy(self):
        cs = CubicSpline(self.edges[::self.stride], self.fe[::self.stride], bc_type='not-a-knot')
        self.amat = cs.c.T


    def compute_distribution(self):
        if self.number_segments == 1:
            self.histogram, self.sim_edges = np.histogram(self.x, bins=self.nbins)
        else:
            hist_dummy, self.sim_edges = np.histogram(self.x, bins=self.nbins, range=self.hist_range)
            self.histogram += (hist_dummy / len(self.x))

    def compute_free_energy(self):
        self.fe_pos =(self.sim_edges[1:] + self.sim_edges[:-1]) / 2
        self.fe_sim = -self.kT * np.log(self.histogram)
        self.fe_sim -= np.min(self.fe_sim)
        self.fe_max = 3

    def plot_fe(self):
        plt.plot(self.fe_pos, self.fe_sim / self.kT, label = 'PMF from sim.')
        plt.plot(self.edges, self.fe / self.kT, label = 'PMF from input')
        plt.ylim(ymax = self.fe_max)
        plt.ylabel('free energy [kT]')
        plt.xlabel('x')
        plt.legend()
        plt.show()

    def save_fe(self):
        array = np.concatenate(
            (self.fe_pos.reshape((self.nbins,1)), 
            self.fe_sim.reshape((self.nbins,1))), 
            axis = 1)
        np.save(self.path_to_save + 'traj_fe', array)

    def integrate_LE(self):
        self.parse_input()
        self.gen_initial_values()
        self.spline_free_energy()
        if self.single_particle:
            if self.single_particle_method == 'RK':
                print('Using runge-kutta integrator')
                for segment in range(self.number_segments):
                    self.x, self.initials = Runge_Kutta_integrator_LE(
                        self.segment_len, 
                        self.dt, 
                        self.mass, 
                        self.gammas, 
                        self.initials, 
                        self.edges, 
                        self.amat)
                    self.compute_distribution()
                    if self.save:
                        np.save(self.path_to_save+'traj_'+str(segment), self.x)
            else:
                print('Using BAOAB integrator')
                for segment in range(self.number_segments):
                    self.x, self.initials = BAOAB(
                        self.segment_len, 
                        self.dt, 
                        self.mass, 
                        self.gammas, 
                        self.initials, 
                        self.edges, 
                        self.amat)
                    self.compute_distribution()
                    if self.save:
                        np.save(self.path_to_save+'traj_'+str(segment), self.x)

        else:
            for segment in range(self.number_segments):
                self.x, self.initials = Runge_Kutta_integrator_GLE(
                    self.segment_len, 
                    self.dt, 
                    self.mass, 
                    self.gammas, 
                    self.couplings,
                    self.initials, 
                    self.edges, 
                    self.amat)
                self.compute_distribution()
                if self.save:
                    np.save(self.path_to_save + 'traj_'+str(segment), self.x)

        self.compute_free_energy()
        if self.plot:
            self.plot_fe()
        if self.save:
                self.save_fe()