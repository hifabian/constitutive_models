from ..ConstitutiveModel import *


class FENE_PTML(ConstitutiveModel):
    name = "FENE-PTML"

    @staticmethod
    def F(trA, L, ndim=3):
        return (L**2-ndim)/(L**2-trA)

    @staticmethod
    def Z(trA, κ, ndim=3):
        return 1-κ+κ*np.sqrt(trA/ndim)

    @staticmethod
    def E(trA, ε0, ndim=3):
        return 4*ε0/(ndim+trA/ndim)
    @staticmethod
    def equation(A, gradU, Wi, β, L, κ, ε0):
        D = 0.5*(gradU+gradU.transpose())
        return ConstitutiveModel.contravariant_derivative(gradU, A) \
            + FENE_PTML.E(np.trace(A), ε0, A.shape[0]) *(D @ A + A @ D) \
            + ( FENE_PTML.F(np.trace(A), L, A.shape[0]) * A - np.eye(*A.shape) ) \
                / (Wi * FENE_PTML.Z(np.trace(A), κ, A.shape[0]))

    @staticmethod
    def stress_tensor(A, gradU, Wi, β, L, κ, ε0):
        return (1-β) / Wi \
            * ( FENE_PTML.F(np.trace(A), L, A.shape[0]) * A - np.eye(*A.shape) ) \
            + β * (gradU+gradU.transpose())