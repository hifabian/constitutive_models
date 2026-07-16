from .ConstitutiveModel import *


class Multimode(ConstitutiveModel):
    name = "Multimode"
    nmodes = None

    def __init__(self, model, Wiq, w):
        """
        Args:
            model (ConstitutiveModel or array): Underlying constitutive model(s)
            Wiq (array): Weissenberg for each mode relative to nominal Weissenberg number
            w (array):   Weighting factors for each mode
        """
        assert(len(w) == len(Wiq)), "Length of Wiq and w must be the same"
        assert(np.all(np.array(w) >= 0)), "Weights must be non-negative"
        assert(np.isclose(np.sum(w), 1.0)), "Weights must be normalized"

        self.nmodes = len(Wiq)
        self.models = model
        if not isinstance(self.models, np.ndarray):
            self.models = np.full(self.nmodes, self.models)
        self.Wiq = Wiq
        self.w = w

    def equation(self, A, gradU, Wi, con_kwargs):
        res = np.empty(A.shape)
        for idx in range(self.nmodes):
            res[idx] = self.models[idx].equation(A[idx], gradU, Wi*self.Wiq[idx], **con_kwargs[idx])
        return res

    def stress_tensor(self, A, gradU, Wi, con_kwargs):
        τ = np.zeros(A.shape[1:])
        for idx in range(self.nmodes):
            τ += self.w[idx] * self.models[idx].stress_tensor(A[idx], gradU, Wi*self.Wiq[idx], **con_kwargs[idx])
        return τ

    def zero_state(self, ndim):
        A = np.zeros([self.nmodes, ndim, ndim])
        for idx in range(self.nmodes):
            A[idx] = self.models[idx].zero_state(ndim)
        return A