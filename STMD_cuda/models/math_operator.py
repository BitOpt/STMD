import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from STMD_cuda.models.create_kernel import create_gamma_kernel, create_inhi_kernel_W2
from STMD_cuda.models.compute_module import CircularTensorList, compute_circularlist_conv, \
    compute_temporal_conv

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.create_gaussian_kernel()

    def create_gaussian_kernel(self):
        radius = self.kernel_size // 2
        x = torch.arange(-radius, radius + 1, 1)
        gauss_kernel = (1 / (self.sigma * (2 * torch.pi) ** 0.5)) * torch.exp(
            -(x ** 2) / (2 * self.sigma ** 2))
        guass_kernel = gauss_kernel / gauss_kernel.sum()  # 归一化
        kernel_2d = guass_kernel.view(1, 1, -1) * guass_kernel.view(1, -1, 1)
        return kernel_2d.to(device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0).to(device)
        padding = self.kernel_size // 2
        return F.conv2d(x, self.kernel.expand(x.size(1), -1, -1, -1), padding=padding,
                        groups=x.size(1))


class GammaDelay(nn.Module):
    """
        GammaDelay Class

        Implements a gamma filter used in the lamina layer of the ESTMD (Elementary Motion Detection) neural network.
        This filter is based on insect physiology and serves to detect moving targets in visual clutter.

        Parameters:
            order (int): Order of the gamma filter. Default is 1.
            tau (float): Time constant of the filter.
            lenKernel (int): Length of the filter kernel. If not provided, it is calculated based on the time constant.
            isRecord (bool): Flag indicating whether to record input history. Default is True.
            isInLoop (bool): Flag indicating whether to cover the point in CircularCell. Default is False.
    """

    def __init__(self, order=1, tau=1, lenKernel=None):
        super(GammaDelay, self).__init__()
        self.order = order
        self.tau = tau
        self.lenKernel = lenKernel
        self.gammaKernel = None
        self.create_flag = False
        self.cellGamma = None
        self.cellGamma_count = 0

    def init_config(self):
        if self.order < 1:
            self.order = 1

        if self.lenKernel is None:
            self.lenKernel = int(np.ceil(self.tau))
            # self.lenKernel = int(np.ceil(3 * self.tau))

        if self.gammaKernel is None:
            self.gammaKernel = create_gamma_kernel(self.order, self.tau, self.lenKernel)

    def process(self, inputData):
        """Forward pass method."""
        # Process input tensor to generate lamina output
        if self.create_flag is False:
            self.cellGamma = torch.zeros((1, self.lenKernel, inputData.shape[2], inputData.shape[3])).to(device)
            self.create_flag = True

        GammaDelayOpt = self.compute_by_conv(inputData)
        return GammaDelayOpt

    def compute_by_conv(self, inputData):
        """
        version 2.0:
        修改了inputData的格式，由(1,1,x,y)->(1,N,x,y)
        输入可能不止一个通道，有可能是多个通道，即多个图像张量矩阵
        """
        if self.cellGamma_count < self.lenKernel:
            self.cellGamma[0, self.cellGamma_count + 1:] = self.cellGamma[0, self.cellGamma_count:-1].clone()
            self.cellGamma[0, 0:inputData.shape[1]] = inputData
            self.cellGamma_count += inputData.shape[1]
        else:
            self.cellGamma[0, 1:] = self.cellGamma[0, :-1].clone()
            self.cellGamma[0, 0:inputData.shape[1]] = inputData

        gammadelayOpt = torch.sum(self.cellGamma * self.gammaKernel, dim=1, keepdim=True)
        return gammadelayOpt


class GammaBandPassFilter(nn.Module):
    def __init__(self):
        super(GammaBandPassFilter, self).__init__()
        self.hGammaDelay1 = GammaDelay(order=2, tau=3)
        self.hGammaDelay2 = GammaDelay(order=3, tau=6)

        self.hGammaDelay1.init_config()
        self.hGammaDelay2.init_config()

    def init_config(self):
        if self.hGammaDelay1.tau >= self.hGammaDelay2.tau:
            self.hGammaDelay2.tau = self.hGammaDelay1.tau + 1

    def process(self, iptMatrix):
        gamma1Output = self.hGammaDelay1.process(iptMatrix)
        gamma2Output = self.hGammaDelay2.process(iptMatrix)

        optMatrix = gamma1Output - gamma2Output
        return optMatrix


class SurroundInhibition(nn.Module):
    def __init__(self, KernelSize=15, Sigma1=1.5, Sigma2=3, e=1.0, rho=0, A=1, B=3):
        super(SurroundInhibition, self).__init__()
        self.KernelSize = KernelSize
        self.Sigma1 = Sigma1
        self.Sigma2 = Sigma2
        self.e = e
        self.rho = rho
        self.A = A
        self.B = B
        self.inhiKernelW2 = None

    def init_config(self):
        self.inhiKernelW2 = create_inhi_kernel_W2(self.KernelSize,
                                                  self.Sigma1,
                                                  self.Sigma2,
                                                  self.e,
                                                  self.rho,
                                                  self.A,
                                                  self.B)  # 添加通道维度

    def process(self, iptMatrix):
        inhiOpt = F.conv2d(iptMatrix, self.inhiKernelW2, padding=self.KernelSize // 2)
        inhiOpt = torch.clamp(inhiOpt, min=0)
        return inhiOpt
