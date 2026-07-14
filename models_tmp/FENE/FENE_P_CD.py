from ..ConstitutiveModel import *


class FENE_P_CD(ConstitutiveModel):
    name = "FENE-P-CD"

    @staticmethod
    def F(trA, L, ndim=3):
        return (L**2-dim)/(L**2-trA)

    @staticmethod
    def Z(trA, κ, ndim=3):
        return 1-κ+κ*np.sqrt(trA/ndim)

    @staticmethod
    def equation(A, gradU, Wi, β, L, κ):
        return ConstitutiveModel.contravariant_derivative(gradU, A) \
            + ( FENE_P_CD.F(np.trace(A), L, A.shape[0]) * A - np.eye(*A.shape) ) \
                / (Wi * FENE_P_CD.Z(np.trace(A), κ, A.shape[0]))

    @staticmethod
    def stress_tensor(A, gradU, Wi, β, L, κ):
        return (1-β) / Wi \
            * ( FENE_P_CD.F(np.trace(A), L, A.shape[0]) * A - np.eye(*A.shape) ) \
            + β * (gradU+gradU.transpose())