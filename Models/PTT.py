from .ConstitutiveModel import *


class PTT(ConstitutiveModel):
    name = "PTT"

    @staticmethod
    def equation(A, gradU, Wi, β, ε):
        return ConstitutiveModel.contravariant_derivative(gradU, A) \
            + np.exp(ε*(np.linalg.trace(A)-A.shape[0]))* (A - np.eye(*A.shape)) / Wi

    @staticmethod
    def stress_tensor(A, gradU, Wi, β, ε):
        return (1-β) / Wi * (A - np.eye(*A.shape)) + β * (gradU+gradU.transpose())