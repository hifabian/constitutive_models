from ..ConstitutiveModel import *


class FENE_PTML(ConstitutiveModel):
    name = "FENE-PTML"

    @staticmethod
    def F(trA, L):
        return 1/(1-trA/L**2)

    @staticmethod
    def Z(trA, κ):
        return 1-κ+κ*np.sqrt(trA/3)

    @staticmethod
    def E(trA, ε0):
        return 4*ε0/(3+trA/3)

    @staticmethod
    def equation(A, gradU, Wi, β, L, κ, ε0):
        D = 0.5*(gradU+gradU.transpose())
        return ConstitutiveModel.contravariant_derivative(gradU, A) \
            + FENE_PTML.E(np.trace(A), ε0) *(D @ A + A @ D) \
            + ( FENE_PTML.F(np.trace(A), L) * A - np.eye(*A.shape) ) \
                / (Wi * FENE_PTML.Z(np.trace(A), κ))

    @staticmethod
    def stress_tensor(A, gradU, Wi, β, L, κ, ε0):
        return (1-β) / Wi \
            * ( FENE_PTML.F(np.trace(A), L) * A - np.eye(*A.shape) ) \
            + β * (gradU+gradU.transpose())