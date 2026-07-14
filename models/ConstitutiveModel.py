import numpy as np
from abc import ABC, abstractmethod


class ConstitutiveModel(ABC):
    @staticmethod
    @abstractmethod
    def equation(A, gradU, Wi, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def stress_tensor(A, gradU, Wi, **kwargs):
        pass

    @staticmethod
    def contravariant_derivative(gradU, A):
        return -gradU.transpose() @ A - A @ gradU


    @staticmethod
    def covariant_derivative(gradU, A):
        return gradU @ A + A @ gradU.transpose()

    @staticmethod
    def zero_state(ndim):
        return np.eye(ndim)