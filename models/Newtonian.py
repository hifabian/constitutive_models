from .ConstitutiveModel import *


class Newtonian(ConstitutiveModel):
    name = "Newtonian"

    @staticmethod
    def equation(A, gradU, Wi):
        return np.zeros(A.shape)

    @staticmethod
    def stress_tensor(A, gradU, Wi):
        return gradU+gradU.transpose()