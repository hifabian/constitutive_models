
from ..ConstitutiveModel import *


class FENE_CR(ConstitutiveModel):
    name = "FENE-CR"

    @staticmethod
    def F(trA, L):
        return 1/(1-trA/L**2)

    @staticmethod
    def equation(A, gradU, Wi, β, L):
        return ConstitutiveModel.contravariant_derivative(gradU, A) \
            + FENE_CR.F(np.trace(A), L) * ( A - np.eye(*A.shape) ) / Wi

    @staticmethod
    def stress_tensor(A, gradU, Wi, β, L):
        return (1-β) / Wi * FENE_CR.F(np.trace(A), L) * ( A - np.eye(*A.shape) ) \
            + β * (gradU+gradU.transpose())