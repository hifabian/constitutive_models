from .ConstitutiveModel import *


class Saramito(ConstitutiveModel):
    name = "Saramito"

    # NOTE: Following Saramito's definition: no half factor in norm!
    #       rheoTool and many others include the half factor.

    # Simple Shear: lim_{Wi -> 0} eta(Wi) = 1+Bi/sqrt{2}
    # [validated using this relation]
    # Bi = Bn / Wi

    @staticmethod
    def equation(τ, gradU, Wi, β, Bn):
        τd = np.linalg.norm((τ-np.trace(τ)/(τ.shape[0])*np.eye(*τ.shape)))+1e-15
        return ConstitutiveModel.contravariant_derivative(gradU, τ) \
               + max(0, (τd-Bn)/τd)*τ / Wi \
               - (1-β)*(gradU+gradU.T)

    @staticmethod
    def stress_tensor(τ, gradU, Wi, β, Bn):
        return τ / Wi + β * (gradU+gradU.T)

    @staticmethod
    def zero_state(ndim):
        return 0*np.eye(ndim)


    @staticmethod
    def steady_guess(gradU, β, Bn):
        γ = gradU + gradU.T
        return 1.01*max(Bn, 1e-3)*γ/np.linalg.norm(γ)


class Bingham(ConstitutiveModel):
    name = "Bingham"

    @staticmethod
    def equation(τ, gradU, Wi, β, Bn):
        γ0 = np.linalg.norm(gradU+gradU.T)
        M = 1e3 # smoothing TODO probably not quite accurate
        f = M*Wi if γ0 == 0 else -np.expm1(-M*Wi*γ0)/γ0
        return τ - (Bn/Wi*f+1)*γ0

    @staticmethod
    def stress_tensor(τ, gradU, Wi, β, Bn):
        return τ

    @staticmethod
    def zero_state(ndim):
        return 0*np.eye(ndim)