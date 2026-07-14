import warnings
import numpy as np
import scipy.optimize as sopt
import utils
from scipy.integrate import RK45, Radau
from algorithms import pseudo_arclength


def steady(ndim, Wimax, constEq, con_kwargs={}, pal_kwargs={}):
    """Steady shear flow for different constitutive models.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    Wimax : float
        Maximal Wi-value.
    constEq : ConstitutiveModel
        Static class for constitutive model.
    con_kwargs : dict
        Parameters for constitutive model.
    pal_kwargs : dict
        Parameters for pseudo-arclength continuation.

    Returns
    -------
    Wi : narray
        Array of Weissenberg numbers.
    τ : narray
        Array of (polymeric and viscous) stress tensors.
    """
    warnings.formatwarning = utils.warning_format

    # Gradient for different shear flows (non-dimensional)
    gradU = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])[:ndim,:ndim]
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


def startup(ndim, Wi, constEq, con_kwargs={}, ODESolver=RK45, tmax=1e4, ttol=None):
    """Startup shear flow for different constitutive models.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
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

    Returns
    -------
    t : narray
        Array of time steps.
    τ : narray
        Array of (polymeric and viscous) stress tensors.
    """
    warnings.formatwarning = utils.warning_format

    # Gradient for different shear flows (non-dimensional)
    gradU = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])[:ndim,:ndim]
    # Initial values
    t0 = 0.0
    A0 = constEq.zero_state(ndim)
    # RHS
    rhs = lambda A: -constEq.equation(A, gradU, Wi, **con_kwargs)
    adj_rhs = lambda t, A: rhs(A.reshape(A0.shape)).flatten()
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


def relaxation(ndim, γ, dt, constEq, con_kwargs={}, ODESolver=RK45, tmax=1e4, ttol=None):
    """Relaxation (step strain) for different constitutive models.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    γ : float
        Step size.
    dt : float
        Size of step using Gaussian.
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

    Returns
    -------
    t : narray
        Array of time steps.
    τ : narray
        Array of (polymeric and viscous) stress tensors.
    """
    warnings.formatwarning = utils.warning_format

    # Gradient for different shear flows (non-dimensional)
    gradU = lambda t: np.array([[0.0, 0.0, 0.0],
                                [γ/np.sqrt(np.pi*dt**2/9)*np.exp(-(t-dt/2)**2/(dt**2/9)), 0.0, 0.0],
                                [0.0, 0.0, 0.0]])[:ndim,:ndim]
    # Initial values
    t0 = 0.0
    A0 = constEq.zero_state(ndim)
    # RHS
    rhs = lambda A, t: -constEq.equation(A, gradU(t), 1, **con_kwargs)
    adj_rhs = lambda t, A: rhs(A.reshape(A0.shape), t).flatten()
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
        τ[idx] = 1*constEq.stress_tensor(A, gradU(t[idx]), 1, **con_kwargs)
    return t, τ


def cessation(ndim, dγ0, constEq, con_kwargs={}, ODESolver=Radau, tmax=1e3, ttol=None):
    """Cessation for different constitutive models.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    dγ0 : float
        Initial shear rate.
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

    Returns
    -------
    t : narray
        Array of time steps.
    τ : narray
        Array of (polymeric and viscous) stress tensors.
    """
    warnings.formatwarning = utils.warning_format

    def solve_initial(dγ0, constEq, A0, con_kwargs={}):
        # Initial gradient before cessation
        gradU0 = np.array([[0.0, 0.0, 0.0],
                           [dγ0, 0.0, 0.0],
                           [0.0, 0.0, 0.0]])[:ndim,:ndim]
        # Residual for initial state
        residual = lambda A, Wi: constEq.equation(A, gradU0, Wi, **con_kwargs)
        adj_residual = lambda A: residual(A.reshape(constEq.zero_state(ndim).shape), 1).flatten()
        return sopt.root(lambda u: adj_residual(u), x0=A0, method='hybr')

    sol = solve_initial(dγ0, constEq, constEq.zero_state(ndim).flatten(), con_kwargs)
    if not sol.success:
        warnings.warn(f"Failed to find initial state for dγ0={dγ0} and {constEq.name}. Trying to decrease dγ0.",
                      RuntimeWarning)
        attempts = 1
        dγr = dγ0 / 16
        sol = solve_initial(dγr, constEq, constEq.zero_state(ndim).flatten(), con_kwargs)
        while attempts < 200 and ((not sol.success and dγr > 1e-1) or dγr < dγ0):
            attempts += 1
            if not sol.success and dγr > 1e-1:
                dγr /= 4
                sol = solve_initial(dγr, constEq, constEq.zero_state(ndim).flatten(), con_kwargs)
            else:
                dγr *= 2
                sol = solve_initial(dγr, constEq, sol.x, con_kwargs)
        if not sol.success or dγr < dγ0:
            raise RuntimeError(f"Failed to find initial state for dγ0={dγ0} and {constEq.name}.")

    # Gradient for cessation (non-dimensional)
    gradU = lambda t: np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]])[:ndim,:ndim]
    # Initial values
    t0 = 0.0
    A0 = sol.x.reshape(constEq.zero_state(ndim).shape)
    # RHS
    rhs = lambda A, t: -constEq.equation(A, gradU(t), 1, **con_kwargs)
    adj_rhs = lambda t, A: rhs(A.reshape(A0.shape), t).flatten()
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
        τ[idx] = 1*constEq.stress_tensor(A, gradU(t[idx]), 1, **con_kwargs)
    return t, τ