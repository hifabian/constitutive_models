import warnings
import numpy as np
from scipy.integrate import RK45
from algorithms import pseudo_arclength


def steady(ndim, b, Wimax, constEq, con_kwargs={}, pal_kwargs={}):
    """Steady extensional flow for different constitutive models.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    b : 0 or 1
        Uniaxial (0) or planar (1) extensional flow.
    Wimax : float
        Maximal Wi-value.
    constEq : ConstitutiveModel
        Static class for constitutive model.
    con_kwargs : dict, optional
        Parameters for constitutive model.
    pal_kwargs : dict, optional
        Parameters for pseudo-arclength continuation.

    Returns
    -------
    Wi : narray
        Array of Weissenberg numbers.
    τ : narray
        Array of (polymeric and viscous) stress tensors.
    """
    # Gradient for different extensional flows (non-dimensional)
    gradU = np.array([[-0.5*(1+b), 0.0, 0.0],
                      [ 0.0,-0.5*(1-b), 0.0],
                      [ 0.0, 0.0, 1]])
    # Residual + residual for arclength
    residual = lambda A, Wi: constEq.equation(A, gradU, Wi, **con_kwargs)
    adj_residual = lambda T: residual(T[:-1].reshape(constEq.zero_state(ndim).shape), T[-1]).flatten()
    # Run pseudo-arclength continuation
    T0 = np.hstack([constEq.zero_state(ndim).flatten(), 1e-5])
    T = pseudo_arclength(adj_residual, T0, Wimax, **pal_kwargs)
    # Transform output
    nSteps = len(T)
    τ = np.empty((nSteps, ndim, ndim))
    Wi = np.empty((nSteps))
    for idx in range(nSteps):
        A = T[idx][:-1].reshape(constEq.zero_state(ndim).shape)
        Wi[idx] = T[idx][-1]
        τ[idx] = Wi[idx]*constEq.stress_tensor(A, gradU, Wi[idx], **con_kwargs)
    return Wi, τ


def step(ndim, b, Wi, constEq, con_kwargs={}, ODESolver=RK45, tmax=1e3, ttol=None, tchange=0.2):
    """Step extensional flow for different constitutive models.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    b : 0 or 1
        Uniaxial (0) or planar (1) extensional flow.
    Wi : float
        Wi-value.
    constEq : ConstitutiveModel
        Static class for constitutive model.
    con_kwargs : dict
        Parameters for constitutive model.
    ODESolver : scipy.integrate.ODESolver
        Solver for the ODE.
    tmax : float
        Maximum time value (early stopping if seemingly converged).
    ttol : float
        Tolerance before assuming steady state is reached.
    tchange : float
        Fraction of total time to switch to extension cessation
        (hard coded 1% of original).

    Returns
    -------
    t : narray
        Array of time steps.
    τ : narray
        Array of (polymeric and viscous) stress tensors.
    """
    # Gradient for different extensional flows (non-dimensional)
    gradU = np.array([[-0.5*(1+b), 0.0, 0.0],
                      [ 0.0,-0.5*(1-b), 0.0],
                      [ 0.0, 0.0, 1]])
    # Off-On function
    time_control = lambda t: 1.0 if t < tmax * tchange else 0.01
    # Initial values
    t0 = 0.0
    A0 = constEq.zero_state(ndim)
    # RHS
    rhs = lambda A, t: -constEq.equation(A, time_control(t)*gradU, Wi, **con_kwargs)
    adj_rhs = lambda t, A: rhs(A.reshape(A0.shape),t).flatten()
    # Run ODE
    sol = ODESolver(adj_rhs, t0, A0.flatten(), tmax, max_step=1)

    tvals = [t0]
    Avals = [A0]
    running = True
    while running:
        # Solve next step
        sol.step()
        tvals.append(sol.t)
        Avals.append(sol.y.reshape(A0.shape))
        # Determine end criterions:
        if  sol.status == "finished" or  sol.status == "failed":
            warnings.warn(f"Ended due to solver status={sol.status}",
                          RuntimeWarning)
            running = False
        elif ttol is not None and np.linalg.norm(Avals[-1]-Avals[-2]) < np.linalg.norm(Avals[-1])*ttol:
            running = False

    # Transform output
    nSteps = len(tvals)
    τ = np.empty((nSteps, ndim, ndim))
    t = np.array(tvals)
    for idx in range(nSteps):
        A = Avals[idx].reshape(A0.shape)
        τ[idx] = Wi*constEq.stress_tensor(A, gradU, Wi, **con_kwargs)
    return t, τ