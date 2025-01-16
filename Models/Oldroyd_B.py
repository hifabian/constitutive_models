from .ConstitutiveModel import *


class Oldroyd_B(ConstitutiveModel):
    name = "Oldroyd-B"

    @staticmethod
    def equation(A, gradU, Wi, β):
        return ConstitutiveModel.contravariant_derivative(gradU, A) + (A - np.eye(*A.shape)) / Wi

    @staticmethod
    def stress_tensor(A, gradU, Wi, β):
        return (1-β) / Wi * (A - np.eye(*A.shape)) + β * (gradU+gradU.transpose())

    @staticmethod
    def uni_ext_N1(Wi, β):
        return 3*(β+(1-β)/((1-2*Wi)*(1+Wi)))


class Rigid_Oldroyd_B(Oldroyd_B):
    name = "Rigid Oldroyd-B"

    @staticmethod
    def equation(A, gradU, Wi, β):
        return Oldroyd_B.equation(A, gradU, Wi, β) + A @ (gradU+gradU.T) @ A

    @staticmethod
    def stress_tensor(A, gradU, Wi, β):
        # NOTE: Unsure about the A @ A term, derived from L tensor
        return (1-β) / Wi * (A@A - np.eye(*A.shape)) + β * (gradU+gradU.transpose())