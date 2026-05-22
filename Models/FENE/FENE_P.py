from ..ConstitutiveModel import *


class FENE_P(ConstitutiveModel):
    name = "FENE-P"

    @staticmethod
    def F(trA, L, ndim=3):
        return (L**2-ndim)/(L**2-trA)

    @staticmethod
    def equation(A, gradU, Wi, β, L):
        return ConstitutiveModel.contravariant_derivative(gradU, A) \
            + ( FENE_P.F(np.trace(A), L, A.shape[0]) * A - np.eye(*A.shape) ) / Wi

    @staticmethod
    def stress_tensor(A, gradU, Wi, β, L):
        return (1-β) / Wi * ( FENE_P.F(np.trace(A), L, A.shape[0]) * A - np.eye(*A.shape) ) \
            + β * (gradU+gradU.transpose())

    @staticmethod
    def N1(Wi, β, L, ndim=3):
        return (1-β)*L**2/Wi*(6**(2/3)*(9*L**2*Wi**4*(1+np.sqrt(1+2/27*L**2/Wi**2)))**(1/3)
                              / ((9*L**2*Wi*(1+np.sqrt(1+2/27*L**2/Wi**2)))**(2/3)-6**(1/3)*L**2)-1)