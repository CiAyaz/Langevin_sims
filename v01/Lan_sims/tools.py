import numpy as np
from numba import njit
from math import sqrt, exp


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
    xi_factor[1] = sqrt(2 * kT * gammas[1] * dt)
    for y in range(2, number_vars):
        xi_factor[y] = sqrt(2 * kT / gammas[y] * dt)
    
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
            k[rk, 0] = dt * vars[rk, 1]
            k[rk, 1] = (dt * (force(vars[rk, 0], amatrix, pot_edges, start_bins, width_bins)
            - gammas[1] * vars[rk, 1]) + xi_factor[1] * xi[1]) / m
            # orhtogonal degrees of freedom
            for y in range(2, number_vars):
                k[rk, 1] += dt * coupling_force(vars[rk, 0], vars[rk, y], couplings[y - 2]) / m
                k[rk, y] = (dt * coupling_force(vars[rk, y], vars[rk, 0], couplings[y - 2]) / gammas[y]
                + xi_factor[y]*xi[y])
                vars[rk + 1, y] = vars[0, y] + RK[rk] * k[rk, y]
            # variable of interest
            vars[rk + 1, 0] = vars[0, 0] + RK[rk] * k[rk, 0]
            vars[rk + 1, 1] = vars[0, 1] + RK[rk] * k[rk, 1]

        # last runge kutta step
        k[3, 0] = dt * vars[3, 1]
        k[3, 1] = (dt * (force(vars[3, 0], amatrix, pot_edges, start_bins, width_bins)
        - gammas[1] * vars[3, 1]) + xi_factor[1] * xi[1]) / m
        # orhtogonal degrees of freedom
        for y in range(2, number_vars):
            k[3, 1] += dt * coupling_force(vars[3, 0], vars[3, y], couplings[y - 2]) / m
            k[3, y] = (dt * coupling_force(vars[3, y], vars[3, 0], couplings[y - 2]) / gammas[y]
            + xi_factor[y]*xi[y])
            vars[0, y] += (k[0, y] + 2 * k[1, y] + 2 * k[2, y] + k[3, y]) / 6
        # variable of interest
        vars[0, 0] += (k[0, 0] + 2 * k[1, 0] + 2 * k[2, 0] + k[3, 0]) / 6
        vars[0, 1] += (k[0, 1] + 2 * k[1, 1] + 2 * k[2, 1] + k[3, 1]) / 6

        x[step] = vars[0, 0]

    return x, vars[0]


@njit()
def Runge_Kutta_integrator_LE(
    nsteps, dt, m, gamma, initials, pot_edges, amatrix, kT=2.494):
    """
    Integrator for a single particle underdamped Langevin eq.
    """

    # relevant constants
    xi_factor = sqrt(2 * kT * gamma * dt)
    
    # runge kutta step factors
    RK = np.array([0.5, 0.5, 1.])

    # arrays to store temp data
    vars = np.zeros((4, 2))
    vars[0] = initials
    k = np.zeros((4, 2))
    
    # trajectory array
    x = np.zeros(nsteps)

    # parameters for spline
    width_bins = pot_edges[1] - pot_edges[0]
    start_bins = pot_edges[0]

    for step in range(nsteps):
        # draw random force
        xi = np.random.normal(0., 1.)
        # first 3 runge kutta steps
        for rk in range(3):
            k[rk, 0] = dt * vars[rk, 1]
            k[rk, 1] = (dt * (force(vars[rk, 0], amatrix, pot_edges, start_bins, width_bins)
            - gamma * vars[rk, 1]) + xi_factor * xi) / m
            vars[rk + 1, 0] = vars[0, 0] + RK[rk] * k[rk, 0]
            vars[rk + 1, 1] = vars[0, 1] + RK[rk] * k[rk, 1]

        # last runge kutta step
        k[3, 0] = dt * vars[3, 1]
        k[3, 1] = (dt * (force(vars[3, 0], amatrix, pot_edges, start_bins, width_bins)
        - gamma * vars[3, 1]) + xi_factor * xi) / m
        vars[0, 0] += (k[0, 0] + 2 * k[1, 0] + 2 * k[2, 0] + k[3, 0]) / 6
        vars[0, 1] += (k[0, 1] + 2 * k[1, 1] + 2 * k[2, 1] + k[3, 1]) / 6

        x[step] = vars[0, 0]

    return x, vars[0]

@njit()
def BAOAB(nsteps, dt, m, gamma, initials, pot_edges, amatrix, kT=2.494):
    """Langevin integrator for initial value problems
    This function implements the BAOAB algorithm of Benedict Leimkuhler
    and Charles Matthews. See J. Chem. Phys. 138, 174102 (2013) for
    further details.
    Arguments:
        force (function): computes the forces of a single configuration
        nsteps (int): number of integration steps
        x_init (numpy.ndarray(n, d)): initial configuration
        v_init (numpy.ndarray(n, d)): initial velocities
        m (numpy.ndarray(n)): particle masses
        dt (float): time step for the integration
        gammas (float): gammas term, use zero if not coupled
        kT (float): thermal energy
    Returns:
        x (numpy.ndarray(nsteps + 1, n, d)): configuraiton trajectory
        v (numpy.ndarray(nsteps + 1, n, d)): velocity trajectory
    """

    th = 0.5 * dt
    thm = 0.5 * dt / m
    edt = exp(-gamma * dt)
    sqf = sqrt((1.0 - edt ** 2) / (m / kT))
    x = np.zeros(nsteps)
    x[0] = initials[0]
    v = initials[1]
    start_bins = pot_edges[0]
    width_bins = pot_edges[1] - pot_edges[0]
    f = force(x[0], amatrix, pot_edges, start_bins, width_bins)
    for i in range(nsteps):
        v += thm * f
        x[i + 1] = x[i] + th * v
        v *= edt 
        v += sqf * np.random.randn()
        x[i + 1] = x[i + 1] + th * v
        f = force(x[i + 1], amatrix, pot_edges, start_bins, width_bins)
        v += thm * f
    initials = np.array([x[-1], v])
    return x, initials