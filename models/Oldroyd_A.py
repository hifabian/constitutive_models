from .ConstitutiveModel import *


class Oldroyd_A(ConstitutiveModel):
    name = "Oldroyd-A"

    @staticmethod
    def equation(A, gradU, Wi, β):
        return ConstitutiveModel.covariant_derivative(gradU, A) + (A - np.eye(*A.shape)) / Wi

    @staticmethod
    def stress_tensor(A, gradU, Wi, β):
        return (1-β) / Wi * (np.eye(*A.shape) - A) + β * (gradU+gradU.transpose())

    @staticmethod
    def uni_ext_N1(Wi, β):
        return 3*(β+(1-β)/((1+2*Wi)*(1-Wi)))


class Rigid_Oldroyd_A(Oldroyd_A):
    name = "Rigid Oldroyd-A"

    @staticmethod
    def equation(A, gradU, Wi, β):
        return Oldroyd_A.equation(A, gradU, Wi, β) - A @ (gradU+gradU.T) @ A

    @staticmethod
    def stress_tensor(A, gradU, Wi, β):
        # NOTE: Unsure about the A @ A term, derived from L tensor
        return (1-β) / Wi * (np.eye(*A.shape) - A@A) + β * (gradU+gradU.transpose())