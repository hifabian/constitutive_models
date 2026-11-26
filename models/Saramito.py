from .ConstitutiveModel import *


class Saramito(ConstitutiveModel):
    name = "Saramito"

    # NOTE: Following Saramito's definition: no half factor in norm!
    #       rheoTool and many others include the half factor.

    # Simple Shear: lim_{Wi -> 0} eta(Wi) = 1+Bi/sqrt{2}
    # [validated using this relation]

    @staticmethod
    def equation(τ, gradU, Wi, β, Bi):
        τd = np.linalg.norm((τ-np.trace(τ)/(τ.shape[0])*np.eye(*τ.shape)))+1e-15
        return ConstitutiveModel.contravariant_derivative(gradU, τ) \
               + max(0, (τd-Bi)/τd)*τ / Wi \
               - (1-β)*(gradU+gradU.T) / Wi

    @staticmethod
    def stress_tensor(τ, gradU, Wi, β, Bi):
        return τ + β * (gradU+gradU.T)

    @staticmethod
    def zero_state(ndim):
        return 0*np.eye(ndim)


class Bingham(ConstitutiveModel):
    name = "Bingham"

    @staticmethod
    def equation(τ, gradU, Wi, β, Bi):
        γ0 = np.linalg.norm(gradU+gradU.T)+1e-15
        return τ - (Bi/γ0 + (1-β))*(gradU+gradU.T)

    @staticmethod
    def stress_tensor(τ, gradU, Wi, β, Bi):
        return τ + β * (gradU+gradU.T)

    @staticmethod
    def zero_state(ndim):
        return 0*np.eye(ndim)