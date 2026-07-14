from .ConstitutiveModel import *


class CoupledUCM(ConstitutiveModel):
    """
    See: B.J. Edwards, A.N. Beris, and V.G. Mavrantzas.
         "A model with two coupled Maxwell modes", J. Rheol. 40, 917-942 (1996).

    Notes
    -----
    Potentially some issues with the exact parameters, but overall correct behavior.

    Parameters
    ----------
    Wi : float
        Characteristic flow rate multiplied by first mode relaxation time.
    λ : float
        Relaxation time ratio of first mode compared to second mode.
    β1 : float
        Viscosity fraction of first mode. Total is given by sum of both modes.
    θ :
        Interaction term.
    """
    name = "2-Mode Coupled UCM"

    @staticmethod
    def equation(A, gradU, Wi, λ, β1, θ):
        I = np.eye(*A.shape[1:])
        res = np.empty(A.shape)
        res[0] =  ConstitutiveModel.contravariant_derivative(gradU, A[0]) \
                + 1 / Wi * (A[0] - I) + θ / np.sqrt(Wi**2/λ) * np.sqrt((1-β1) / β1) \
                    * (0.5*(A[0]@A[1]+A[1]@A[0]) - A[0])
        res[1] =  ConstitutiveModel.contravariant_derivative(gradU, A[1]) \
                + λ / Wi * (A[1] - I) + θ / np.sqrt(Wi**2/λ) * np.sqrt(β1 / (1-β1)) \
                    * (0.5*(A[0]@A[1]+A[1]@A[0]) - A[1])
        return res

    @staticmethod
    def stress_tensor(A, gradU, Wi, λ, β1, θ):
        τ = np.zeros(A.shape[1:])
        τ += β1       / Wi * (A[0] - np.eye(*τ.shape))
        τ += (1-β1)*λ / Wi * (A[1] - np.eye(*τ.shape))
        return τ

    def zero_state(ndim):
        return np.tile(np.eye(ndim), (2,1,1))