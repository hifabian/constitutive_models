from .ConstitutiveModel import *


class Saramito(ConstitutiveModel):
    name = "Saramito"

    # NOTE: Following Saramito's definition: no half factor in norm!
    #       rheoTool and many others include the half factor.

    # Simple Shear: lim_{Wi -> 0} eta(Wi) = 1+Bi/sqrt{2}
    # [validated using this relation]

    @staticmethod
    def equation(τ, gradU, Wi, β, Bi):
        τd = np.linalg.norm((τ-np.trace(τ)/(len(τ.shape))*np.eye(*τ.shape)))+1e-15
        return Wi*ConstitutiveModel.covariant_derivative(gradU, τ) \
               + max(0, (τd-Bi)/τd)*τ \
               - (1-β)*(gradU+gradU.transpose())

    @staticmethod
    def stress_tensor(τ, gradU, Wi, β, Bi):
        return τ + β * (gradU+gradU.transpose())

    @staticmethod
    def zero_state(ndim):
        return 0*np.eye(ndim)


class Bingham(ConstitutiveModel):
    name = "Bingham"

    # TODO this is probably wrong; needs minimum apparent viscosity (i.e., regularization)

    @staticmethod
    def equation(τ, gradU, Wi, β, Bi):
        γ0 = np.linalg.norm(gradU+gradU.transpose())+1e-15
        τt = Bingham.stress_tensor(τ, gradU, Wi, β, Bi)
        τd = np.linalg.norm((τt-np.trace(τt)/(len(τ.shape))*np.eye(*τ.shape)))
        return 0*τ if τd < 1e-15 else τ - min(0, Bi/γ0 + (1-β))*(gradU+gradU.transpose())

    @staticmethod
    def stress_tensor(τ, gradU, Wi, β, Bi):
        return τ + β * (gradU+gradU.transpose())

    @staticmethod
    def zero_state(ndim):
        return 0*np.eye(ndim)