from .ConstitutiveModel import *


class Corotational_Maxwell(ConstitutiveModel):
    name = "Corotational Maxwell"

    @staticmethod
    def equation(τ, gradU, Wi):
        return Wi*(0.5*ConstitutiveModel.contravariant_derivative(gradU, τ) \
                 + 0.5*(gradU @ τ + τ @ gradU.transpose())) \
            + (τ - (gradU+gradU.transpose()))

    @staticmethod
    def stress_tensor(τ, gradU, Wi):
        return τ