import numpy as np
import scipy.optimize as sopt
import warnings


def pseudo_arclength(residual, T0, λmax, maxIt=2**15, ds0=1e-2, dsTtol=1e-2, dsmin=1e-5, dsmax=1e3):
    """
    Notes
    -----
    The direction vectors are updated using the approximation:
        dT1 = (T1 - T0) / ds

    Parameters
    ----------
    residual : callable
        A vector function to find root of.
    T0 : ndarray
        Initial guess, containing λ in final position.
    λmax : float
        Maximal λ-value for iteration.
    maxIt : int, optional
        Maximum number of iterations.
    ds0 : float, optional
        Initial stepsize.
    dsTtol : float, optional
        Relative tolerance for increasing stepsize.
    dsmin : float, optional
        Minimal stepsize.
    dsmax : float, optional
        Maximal stepsize.

    Returns
    -------
    T : list of ndarray
        Trajectory in solution space.
    """

    tres = lambda T1, T0, dT0, ds: np.hstack([residual(T1), dT0 @ (T1-T0) - ds])
    ds = ds0
    # Calculate: T0 (potentially redundant)
    sol = sopt.root(lambda u: residual(np.hstack([u, T0[-1]])), x0=T0[:-1])
    T0 = np.hstack([sol.x, T0[-1]])
    T = [T0]
    # Calculate: T1
    sol = sopt.root(lambda u: residual(np.hstack([u, T0[-1]+ds])), x0=T0[:-1])
    T1 = np.hstack([sol.x, T0[-1]+ds])
    T.extend([T1])
    # Estimate: dT1
    dT0 = (T1-T0) / ds
    dT0 /= np.linalg.norm(dT0)

    ## Loop until done or failure
    it = 0
    while T[-1][-1] < λmax and it < maxIt:
        it += 1
        success = False
        while not success and ds >= dsmin:
            # Solve system
            T0 = T[-1]
            sol = sopt.root(lambda T1: tres(T1, T0, dT0, ds),
                            x0=T0, options={"maxfev": 1000})
            T1 = sol.x
            success = sol.success

            if success:  # Success
                # Update arclength gradient
                dT0 = (T1-T0) / ds  # Good enough
                dT0 /= np.linalg.norm(dT0)
                # Update stepsize
                if np.linalg.norm(T1-T0) < dsTtol*np.linalg.norm(T1):
                    ds = min(dsmax, ds*1.5)
            else:
                ds /= 2

        # Failure due to stepsize
        if ds < dsmin:
            warnings.warn(f"Reached minimum ds size: {ds}. Aborting loop at λ={T[-1][-1]}",
                          RuntimeWarning)
            break

        T.extend([T1])
    return T