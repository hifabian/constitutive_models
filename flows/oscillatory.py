import warnings
import numpy as np
import scipy.optimize as sopt
from scipy.integrate import odeint, RK45, Radau
from algorithms import pseudo_arclength


def shear(ndim, Wi, constEq, dγdt, t, con_kwargs={}):
    """Oscillatory shear for different constitutive models.

    Can be used for small amplitude oscillatory shear (SAOS), large amplitude
    oscillatory shear (LAOS), or chirp rheology.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    Wi : float
        Wi-value, corresponding to relaxation time here.
    constEq : ConstitutiveModel
        Static class for constitutive model.
    dγdt : lambda(t)
        Shear rate function driving the flow.
    t : numpy.array
        Time sequence of solution.
    con_kwargs : dict
        Parameters for constitutive model.

    Returns
    -------
    t : narray
        Array of time steps.
    τ : narray
        Array of (polymeric and viscous) stress tensors.
    """
    # Gradient for different shear flows (non-dimensional)
    gradU = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])[:ndim,:ndim]
    # Initial values
    A0 = constEq.zero_state(ndim)
    # RHS
    rhs = lambda t, A: -constEq.equation(A, gradU*dγdt(t), Wi, **con_kwargs)
    adj_rhs = lambda A, t: rhs(t, A.reshape(A0.shape)).flatten()
    # Run ODE
    Avals = odeint(adj_rhs, A0.flatten(), t)

    # Transform output
    nSteps = len(t)
    τ = np.empty((nSteps, ndim, ndim))
    for idx in range(nSteps):
        A = Avals[idx].reshape(A0.shape)
        τ[idx] = constEq.stress_tensor(A, gradU*dγdt(t[idx]), Wi, **con_kwargs)
    return τ