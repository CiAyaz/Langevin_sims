import numpy as np
from numba import njit


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
    idx = min(idx, len(pot_edges) - 1)

    # evaluate the gradient of the spline rep
    output = -(
        3 * amatrix[idx, 0] * (x - pot_edges[idx]) ** 2
        + 2 * amatrix[idx, 1] * (x - pot_edges[idx])
        + amatrix[idx, 2]
    )
    return output


@njit()
def integrate_LE(
    nsteps, dt, m, gammas, tgammas, L, x0, v, R, kT, pot_edges, amatrix
):
    """
    Integrator for a Markovian Embedding with exponentially
    decaying memory kernels, characterized by friction gamma[i]
    and memory time tgammas[i]. Uses spline rep from pot_edges
    and amatrix. Restarts sim from pos x0 and velocitiy v
    and position of the overdamped orth. dof at R.
    """

    # relevant constants
    nr_exps = len(gammas)
    gamma = gammas.sum()
    td = L ** 2 * gamma / kT
    tm = m / (gamma * td)
    g = gammas / gamma
    xi_sigma = np.sqrt(2 * g)
    tg = tgammas / td
    xi_factor = np.sqrt(1 / dt)

    # arrays to store temp data
    xi = np.zeros(nr_exps)
    kR = np.zeros((nr_exps, 4))
    Rtemp = np.zeros(nr_exps)
    # trajectory array
    x = np.zeros(nsteps)
    current_pos_x = x0 / L

    # parameters for spline
    pot_edges = np.copy(pot_edges / L)
    width_bins = np.mean(pot_edges[1:] - pot_edges[:-1])
    start_bins = pot_edges[0]

    for step in range(nsteps):
        # first runge kutta step
        # orhtogonal degrees of freedom
        Rsum = 0
        for i in range(nr_exps):
            xi[i] = xi_factor * np.random.normal(loc=0.0, scale=xi_sigma[i])
            Rsum += g[i] * (R[i] - current_pos_x) / tg[i]
        # variable of interest
        kx1 = dt * v
        kv1 = (
            dt
            / tm
            * (Rsum + force(current_pos_x, amatrix, pot_edges, start_bins, width_bins))
        )

        for i in range(nr_exps):
            kR[i, 0] = -dt * ((R[i] - current_pos_x) / tg[i] - xi[i] / g[i])
            Rtemp[i] = R[i] + kR[i, 0] / 2
        x1 = current_pos_x + kx1 / 2
        v1 = v + kv1 / 2

        # second runge kutta step
        Rsum = 0
        for i in range(nr_exps):
            Rsum += g[i] * (Rtemp[i] - x1) / tg[i]
        kx2 = dt * v1
        kv2 = dt / tm * (Rsum + force(x1, amatrix, pot_edges, start_bins, width_bins))

        for i in range(nr_exps):
            kR[i, 1] = -dt * ((Rtemp[i] - x1) / tg[i] - xi[i] / g[i])
            Rtemp[i] = R[i] + kR[i, 1] / 2
        x2 = current_pos_x + kx2 / 2
        v2 = v + kv2 / 2

        # third kutta step
        Rsum = 0
        for i in range(nr_exps):
            Rsum += g[i] * (Rtemp[i] - x2) / tg[i]
        kx3 = dt * v2
        kv3 = dt / tm * (Rsum + force(x2, amatrix, pot_edges, start_bins, width_bins))

        for i in range(nr_exps):
            kR[i, 2] = -dt * ((Rtemp[i] - x2) / tg[i] - xi[i] / g[i])
            Rtemp[i] = R[i] + kR[i, 2]
        x3 = current_pos_x + kx3
        v3 = v + kv3

        # fourth runge kutta step
        Rsum = 0
        for i in range(nr_exps):
            Rsum += g[i] * (Rtemp[i] - x3) / tg[i]
        kx4 = dt * v3
        kv4 = dt / tm * (Rsum + force(x3, amatrix, pot_edges, start_bins, width_bins))

        for i in range(nr_exps):
            kR[i, 3] = -dt * ((Rtemp[i] - x3) / tg[i] - xi[i] / g[i])
            Rtemp[i] = R[i] + kR[i, 3]

        # join all steps
        current_pos_x += (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6
        v += (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6

        # update data arrays
        for i in range(nr_exps):
            R[i] += (kR[i, 0] + 2 * kR[i, 1] + 2 * kR[i, 2] + kR[i, 3]) / 6
        x[step] = current_pos_x

    final = np.array([x[-1] * L, v])
    return x * L, final, R
