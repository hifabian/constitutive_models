
from ..ConstitutiveModel import *


class FENE_CR(ConstitutiveModel):
    name = "FENE-CR"

    @staticmethod
    def F(trA, L, ndim=3):
        return (L**2-ndim)/(L**2-trA)

    @staticmethod
    def equation(A, gradU, Wi, β, L):
        return ConstitutiveModel.contravariant_derivative(gradU, A) \
            + FENE_CR.F(np.trace(A), L, A.shape[0]) * ( A - np.eye(*A.shape) ) / Wi

    @staticmethod
    def stress_tensor(A, gradU, Wi, β, L):
        return (1-β) / Wi * ( FENE_CR.F(np.trace(A), L, A.shape[0]) * A - np.eye(*A.shape) ) \
            + β * (gradU+gradU.transpose())

    @staticmethod
    def N1(Wi, β, L, ndim=3):
        return (1-β)/(2*Wi) * (L**2-ndim) * (np.sqrt(1+ 8*Wi**2/(L**2-ndim))-1)