from ..ConstitutiveModel import *
from .FENE_P import *


class FENE_P_Multimode(ConstitutiveModel):
    name = "FENE-P Multimode"
    nmodes = None

    def __init__(cls, n):
        cls.nmodes = n

    def equation(cls, A, gradU, Wi, β, L, λfrac):
        res = np.empty(A.shape)
        for idx in range(cls.nmodes):
            res[idx] = ConstitutiveModel.contravariant_derivative(gradU, A[idx]) \
            + ( FENE_P.F(np.trace(A[idx]), L[idx]) * A[idx] - np.eye(*A.shape[1:]) ) / (Wi*λfrac[idx])
        return res

    def stress_tensor(cls, A, gradU, Wi, β, L, λfrac):
        τ = np.zeros(A.shape[1:])
        for idx in range(cls.nmodes):
            τ += β[idx] / (Wi*λfrac[idx]) * ( FENE_P.F(np.trace(A[idx]), L[idx]) * A[idx] - np.eye(*τ.shape) )
        return τ + (1-np.sum(β)) * (gradU+gradU.transpose())

    def zero_state(cls, ndim):
        return np.tile(np.eye(ndim), (cls.nmodes,1,1))