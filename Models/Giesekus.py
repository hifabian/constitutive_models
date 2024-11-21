from .ConstitutiveModel import *


class Giesekus(ConstitutiveModel):
    name = "Giesekus"

    @staticmethod
    def equation(A, gradU, Wi, β, α):
        return ConstitutiveModel.contravariant_derivative(gradU, A) \
            + ( (1-α)*np.eye(*A.shape)+α*A ) @ ( A - np.eye(*A.shape) ) / Wi

    @staticmethod
    def stress_tensor(A, gradU, Wi, β, α):
        return (1-β) / Wi * (A - np.eye(*A.shape)) \
            + β * (gradU+gradU.transpose())