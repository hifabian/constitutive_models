from .ConstitutiveModel import *


class Corotational_Maxwell(ConstitutiveModel):
    name = "Corotational Maxwell"

    @staticmethod
    def equation(τ, gradU, Wi):
        return 0.5*ConstitutiveModel.contravariant_derivative(gradU, τ) \
            + 0.5*(gradU @ τ + τ @ gradU.T) \
            + (τ - (gradU+gradU.T)) / Wi

    @staticmethod
    def stress_tensor(τ, gradU, Wi):
        return τ