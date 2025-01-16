from .ConstitutiveModel import *


class Johnson_Segalman(ConstitutiveModel):
    name = "Johnson-Segalman"

    @staticmethod
    def equation(A, gradU, Wi, β, ε0):
        D = 0.5*(gradU+gradU.transpose())
        return ConstitutiveModel.contravariant_derivative(gradU, A) \
            + ε0*(D @ A + A @ D) \
            + (A - np.eye(*A.shape)) / Wi

    @staticmethod
    def stress_tensor(A, gradU, Wi, β, ε0):
        return (1-β) / ((1-ε0)*Wi) * (A - np.eye(*A.shape)) + β * (gradU+gradU.transpose())



class Rigid_Johnson_Segalman(Johnson_Segalman):
    name = "Rigid Johnson-Segalman"

    @staticmethod
    def equation(A, gradU, Wi, β, ε0):
        return Johnson_Segalman.equation(A, gradU, Wi, β, ε0) + (1-ε0)*A @ (gradU+gradU.T) @ A

    @staticmethod
    def stress_tensor(A, gradU, Wi, β, ε0):
        return (1-β) / ((1-ε0)*Wi) * (A@A - np.eye(*A.shape)) + β * (gradU+gradU.transpose())