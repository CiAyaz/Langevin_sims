import numpy as np
from numba import njit
from math import sqrt

from pyrsistent import v
from regex import V0


@njit
def force(x, amatrix, pot_edges, start_bins, width_bins):
    """ Evaluates the spline representation of
        the potential at x, returns the negativ
        gradient of the splines. Takes the
        spline coefficients from the upper
        namespace. """

    # find index of bin by shifting by the start of
    # the first bin and floor dividing
    idx = int((x - start_bins) // width_bins)
    idx = max(idx, 0)
    idx = min(idx, len(pot_edges) - 2)

    # evaluate the gradient of the spline rep
    output = -(
        3 * amatrix[idx, 0] * (x - pot_edges[idx]) ** 2
        + 2 * amatrix[idx, 1] * (x - pot_edges[idx])
        + amatrix[idx, 2]
    )
    return output

@njit
def coupling_force(x, y, coupling_k):
    return - coupling_k * (x - y)

@njit()
def Runge_Kutta_integrator_GLE(
    nsteps, dt, m, gammas, couplings, initials, pot_edges, amatrix, kT=2.494):
    """
    Integrator for a Markovian Embedding with exponentially
    decaying memory kernels, characterized by friction gamma[i]
    and memory time tgammas[i]. Uses spline rep from pot_edges
    and amatrix. Restarts sim from pos x0 and velocitiy v
    and position of the overdamped orth. dof at R.
    """

    # relevant constants
    number_vars = len(gammas)
    xi_factor = np.zeros(number_vars)
    xi = np.zeros(number_vars)
    xi_factor[1] = sqrt(2 * kT * gammas[1] / dt)
    for y in range(2, number_vars):
        xi_factor[y] = sqrt(2 * kT / gammas[y] / dt)
    
    # runge kutta step factors
    RK = np.array([0.5, 0.5, 1.])

    # arrays to store temp data
    vars = np.zeros((4, number_vars))
    vars[0] = initials
    k = np.zeros((4, number_vars))
    
    # trajectory array
    x = np.zeros(nsteps)

    # parameters for spline
    width_bins = pot_edges[1] - pot_edges[0]
    start_bins = pot_edges[0]

    for step in range(nsteps):
        # draw random force
        xi[1:] = np.random.normal(0., 1., number_vars - 1)
        # first 3 runge kutta steps
        for rk in range(3):
            k[rk, 0] = vars[rk, 0]
            k[rk, 1] = (force(vars[rk, 0], amatrix, pot_edges, start_bins, width_bins)
            - gammas[1] * vars[rk, 1] + xi_factor[1] * xi[1]) / m
            # orhtogonal degrees of freedom
            for y in range(2, number_vars):
                k[rk, 1] += coupling_force(vars[rk, 0], vars[rk, y], couplings[y - 2]) / m
                k[rk, y] = (coupling_force(vars[rk, y], vars[rk, 0], couplings[y - 2]) / gammas[y]
                + xi_factor[y]*xi[y])
                vars[rk + 1, y] = vars[0, y] + RK[rk] * dt * k[rk, y]
            # variable of interest
            vars[rk + 1, 0] = vars[0, 0] + RK[rk] * dt * k[rk, 0]
            vars[rk + 1, 1] = vars[0, 1] + RK[rk] * dt * k[rk, 1]

        # last runge kutta step
        k[3, 0] = vars[3, 0]
        k[3, 1] = (force(vars[3, 0], amatrix, pot_edges, start_bins, width_bins)
        - gammas[1] * vars[3, 1] + xi_factor[1] * xi[1]) / m
        # orhtogonal degrees of freedom
        for y in range(2, number_vars):
            k[3, 1] += coupling_force(vars[3, 0], vars[3, y], couplings[y - 2]) / m
            k[3, y] = (coupling_force(vars[3, y], vars[3, 0], couplings[y - 2]) / gammas[y]
            + xi_factor[y]*xi[y])
            vars[0, y] += dt * (k[0, y] + 0.5 * k[1, y] + 0.5 * k[2, y] + k[3, y]) / 6
        # variable of interest
        vars[0, 0] += dt * (k[0, 0] + 0.5 * k[1, 0] + 0.5 * k[2, 0] + k[3, 0]) / 6
        vars[0, 1] += dt * (k[0, 1] + 0.5 * k[1, 1] + 0.5 * k[2, 1] + k[3, 1]) / 6

        x[step] = vars[0, 0]

    return x, vars[0]
