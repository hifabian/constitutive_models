from .ConstitutiveModel import *


class FTQ(ConstitutiveModel):
    name = "Folgar-Tucker with Quadratic Closure"

    # TODO only for infinite dumbbells: no Bretherton and only Giesekus stress

    @staticmethod
    def equation(A, gradU, Pe):
        return (ConstitutiveModel.contravariant_derivative(gradU, A) + np.trace(A@(gradU+gradU.T))*A) + 6*(A - np.eye(*A.shape)/3) / Pe

    @staticmethod
    def stress_tensor(A, gradU, Pe):
        # Unsure why, but need to divide by Pe and not multiply for correct non-dimensionalization
        return 3*(A - np.eye(*A.shape)/3) / Pe + 0.5*np.trace(A@(gradU))*A
