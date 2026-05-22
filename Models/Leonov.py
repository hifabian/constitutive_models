from .ConstitutiveModel import *


class Leonov(ConstitutiveModel):
    name = "Leonov"
    altname = ["Zurich-Rheology"]

    # NOTE: Following the definition of Beris and Edwards (1990, 1994).
    #       This definition is thermodynamically consistent.

    # NOTE: The stress tensor may differ by an isotropic contribution.

    # NOTE: For n=2, this model reduces to the Giesekus model with α=1/2.

    @staticmethod
    def constraint(A):
        return np.linalg.det(A) - 1.0

    @staticmethod
    def equation(A, gradU, Wi, β):
        return ConstitutiveModel.contravariant_derivative(gradU, A) + \
            (A - np.trace(A)/A.shape[0]*np.eye(*A.shape)) @ A / Wi

    @staticmethod
    def stress_tensor(A, gradU, Wi, β):
        return (1-β) / Wi * A + β * (gradU+gradU.transpose())