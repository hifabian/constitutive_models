from .ConstitutiveModel import *


class UCM(ConstitutiveModel):
    name = "UCM"

    @staticmethod
    def equation(A, gradU, Wi):
        return ConstitutiveModel.contravariant_derivative(gradU, A) + (A - np.eye(*A.shape)) / Wi

    @staticmethod
    def stress_tensor(A, gradU, Wi):
        return 1 / Wi * (A - np.eye(*A.shape))

    @staticmethod
    def uni_ext_N1(Wi):
        return 3/((1-2*Wi)*(1+Wi))