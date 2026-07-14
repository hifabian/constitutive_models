from ..ConstitutiveModel import *


class FENE_CD(ConstitutiveModel):
    name = "FENE-CD"

    @staticmethod
    def F(trA, L, ndim=3):
        return (L**2-ndim)/(L**2-trA)

    @staticmethod
    def Z(trA, κ, ndim=3):
        return 1-κ+κ*np.sqrt(trA/ndim)

    @staticmethod
    def equation(A, gradU, Wi, β, L, κ):
        return ConstitutiveModel.contravariant_derivative(gradU, A) \
            + FENE_CD.F(np.trace(A), L, A.shape[0]) * ( A - np.eye(*A.shape) ) \
                / (Wi * FENE_CD.Z(np.trace(A), κ, A.shape[0]))

    @staticmethod
    def stress_tensor(A, gradU, Wi, β, L, κ):
        return (1-β) / Wi \
            * FENE_CD.F(np.trace(A), L, A.shape[0]) * ( A - np.eye(*A.shape) ) \
            + β * (gradU+gradU.transpose())