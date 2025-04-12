import torch
from torch import nn
from torch.nn import functional as F
from STMD_cuda.models.create_kernel import create_fracdiff_kernel
from STMD_cuda.models.compute_module import CircularTensorList, compute_circularlist_conv
from STMD_cuda.models.math_operator import GaussianBlur, SurroundInhibition
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Retina(nn.Module):
    def __init__(self):
        super(Retina, self).__init__()
        self.hGaussianBlur = GaussianBlur(kernel_size=3, sigma=1.0)
    def init_config(self):
        pass

    def forward(self, retinaIpt):
        """
        Processing method.
        Applies the Gaussian blur filter to the input matrix.

        Parameters:
        - retinaIpt: Input matrix.

        Returns:
        - retinaOpt: Output matrix after applying the Gaussian blur filter.
        """
        retinaOpt = self.hGaussianBlur(retinaIpt)
        return retinaOpt


class Lamina(nn.Module):
    """Lamina class for the lamina layer."""

    def __init__(self, alpha=0.8, delta=20):
        """Constructor method."""
        super(Lamina, self).__init__()
        self.alpha = alpha
        self.delta = delta

        # init
        self.fracKernel = create_fracdiff_kernel(self.alpha, self.delta)
        self.paraCur = self.fracKernel[0]
        self.paraPre = np.exp(-self.alpha / (1 - self.alpha))
        self.preLaminaIpt = None
        self.preLaminaOpt = None
        self.create_flag = False
        self.cellRetina = None
        self.cellRetina_count = 0

    def init_config(self):
        pass

    def forward(self, LaminaIpt):
        """Forward pass method."""
        # Process input tensor to generate lamina output
        if self.create_flag is False:
            self.cellRetina = torch.zeros(
                (1, self.delta, LaminaIpt.shape[2], LaminaIpt.shape[3])).to(device)
            self.create_flag = True
        if self.preLaminaIpt is None:
            diffLaminaIpt = torch.zeros_like(LaminaIpt).to(device)
        else:
            diffLaminaIpt = LaminaIpt - self.preLaminaIpt

        laminaOpt = self.compute_by_conv(diffLaminaIpt)
        # laminaOpt = self.compute_by_iteration(diffLaminaIpt)

        self.preLaminaIpt = LaminaIpt

        return laminaOpt

    def compute_by_conv(self, diffLaminaIpt):
        if self.cellRetina_count < self.cellRetina.shape[1]:
            self.cellRetina[0, self.cellRetina_count + 1:] = self.cellRetina[0,
                                                             self.cellRetina_count:-1].clone()
            self.cellRetina[0, 0] = diffLaminaIpt
            self.cellRetina_count += 1
        else:
            self.cellRetina[0, 1:] = self.cellRetina[0, :-1].clone()
            self.cellRetina[0, 0] = diffLaminaIpt

        laminaOpt = torch.sum(self.cellRetina * self.fracKernel, dim=1, keepdim=True)
        return laminaOpt

    '''
    该函数有问题，还未修改，禁止调用
    '''

    def compute_by_iteration(self, diffLaminaIpt):
        """Computes the lamina output by iteration."""
        if self.preLaminaOpt is None:
            laminaOpt = self.paraCur * diffLaminaIpt
        else:
            laminaOpt = self.paraCur * diffLaminaIpt + self.paraPre * self.preLaminaOpt

        self.preLaminaOpt = laminaOpt
        return laminaOpt


class MECumulativeCell(nn.Module):
    def __init__(self):
        super(MECumulativeCell, self).__init__()
        self.gLeak = 0.5
        self.vRest = 0
        self.vExci = 1
        self.postMP = None

    def process(self, samePolarity, oppoPolarity):
        if self.postMP is None:
            self.postMP = torch.zeros_like(samePolarity).to(device)

        # Decay
        decayTerm = self.gLeak * (self.vRest - self.postMP)
        # Inhibition
        inhiGain = 1 + oppoPolarity + oppoPolarity ** 2
        # Excitation
        exciTerm = samePolarity * (self.vExci - self.postMP)

        # Euler method for solving ordinary differential equation
        self.postMP = self.postMP + inhiGain * decayTerm + exciTerm
        return self.postMP


class Medulla(nn.Module):
    def __init__(self):
        super(Medulla, self).__init__()
        self.hMi4 = MECumulativeCell()
        self.hTm9 = MECumulativeCell()

    def init_config(self):
        pass

    def forward(self, medullaIpt):
        """
        Process the input through the Medulla layer.
        Args:
        - medullaIpt (tensor): Input to the Medulla layer.
        Returns:
        - onSignal (tensor): Output signal from Mi4.
        - offSignal (tensor): Output signal from Tm9.
        """
        onSignal = torch.clamp(medullaIpt, min=0)
        offSignal = torch.clamp(-medullaIpt, min=0)

        mi4Opt = self.hMi4.process(onSignal, offSignal)
        tm9Opt = self.hTm9.process(offSignal, onSignal)

        self.Opt = (onSignal, tm9Opt)
        return onSignal, tm9Opt


class LPTangentialCell(nn.Module):
    def __init__(self, kernelSize=3):
        super(LPTangentialCell, self).__init__()
        self.kernelSize = kernelSize
        self.lptcMatrix = None

    def init_config(self):
        self.kernelCos = torch.zeros((self.kernelSize, self.kernelSize))
        self.kernelSin = torch.zeros((self.kernelSize, self.kernelSize))

        halfKernel = self.kernelSize // 2
        for x in range(-halfKernel, halfKernel + 1):
            for y in range(-halfKernel, halfKernel + 1):
                r = (x ** 2 + y ** 2) ** 0.5
                if r == 0:
                    continue
                # 计算cosine值
                self.kernelCos[y + halfKernel, x + halfKernel] = x / r
                self.kernelSin[y + halfKernel, x + halfKernel] = -y / r

        # 将 kernel 转为可训练的参数
        self.kernelCos = self.kernelCos.unsqueeze(0).unsqueeze(0).to(device)
        self.kernelSin = self.kernelSin.unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, laminaOpt: torch.Tensor, onSignal: torch.Tensor, offSignal: torch.Tensor):
        # get indexes
        onDire = (onSignal > 0) & (laminaOpt > 0)
        offDire = (offSignal > 0) & (laminaOpt < 0)

        direMatrix = torch.zeros_like(laminaOpt).to(device)
        # On direction
        direMatrix[onDire] = offSignal[onDire] / onSignal[onDire]
        # Off direction
        direMatrix[offDire] = onSignal[offDire] / offSignal[offDire]

        directionCos = F.conv2d(direMatrix, self.kernelCos,
                                padding=self.kernelSize // 2)
        directionSin = F.conv2d(direMatrix, self.kernelSin,
                                padding=self.kernelSize // 2)

        # 计算方向
        direction = torch.atan2(directionSin, directionCos)
        # 调整方向到 [0, 2*pi] 范围
        direction[direction < 0] += 2 * np.pi

        self.lptcMatrix = direMatrix
        self.Opt = {'direction': direction,
                    'lptcMatric': direMatrix}
        return direction


class Lobula(nn.Module):
    """
    Lobula layer of the motion detection system.
    """

    def __init__(self):
        """
        Initialization method.
        """
        super(Lobula, self).__init__()
        self.hSubInhi = SurroundInhibition()
        self.hLPTC = LPTangentialCell()

    def init_config(self):
        self.hSubInhi.init_config()
        self.hLPTC.init_config()

    def forward(self, onSignal: torch.Tensor, offSignal: torch.Tensor, laminaOpt: torch.Tensor):
        """
        Processing method.
        Args:
        - onSignal (Tensor):  ON channel signal from medulla layer.
        - offSignal (Tensor):  OFF channel signal from medulla layer.
        - laminaOpt (Tensor):  output signal from medulla layer.
        Returns:
        - lobulaOpt (Tensor): output for location.
        - direction (Tensor): output for direction.
        - correlationOutput (Tensor): output without inhibition.
        """
        # direction = self.hLPTC(laminaOpt, onSignal, offSignal)
        direction = None
        correlationOutput = onSignal * offSignal
        lobulaOpt = self.hSubInhi.process(correlationOutput)
        self.Opt = lobulaOpt, direction, correlationOutput
        return lobulaOpt, direction, correlationOutput
