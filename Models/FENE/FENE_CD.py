from ..ConstitutiveModel import *


class FENE_CD(ConstitutiveModel):
    name = "FENE-CD"

    @staticmethod
    def F(trA, L):
        return 1/(1-trA/L**2)

    @staticmethod
    def Z(trA, κ):
        return 1-κ+κ*np.sqrt(trA/3)

    @staticmethod
    def equation(A, gradU, Wi, β, L, κ):
        return ConstitutiveModel.contravariant_derivative(gradU, A) \
            + FENE_CD.F(np.trace(A), L) * ( A - np.eye(*A.shape) ) \
                / (Wi * FENE_CD.Z(np.trace(A), κ))

    @staticmethod
    def stress_tensor(A, gradU, Wi, β, L, κ):
        return (1-β) / Wi \
            * FENE_CD.F(np.trace(A), L) * ( A - np.eye(*A.shape) ) \
            + β * (gradU+gradU.transpose())