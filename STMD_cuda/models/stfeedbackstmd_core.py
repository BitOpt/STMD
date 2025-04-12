import torch
from torch import nn
from torch.nn import functional as F
from STMD_cuda.models.math_operator import GammaDelay, GammaBandPassFilter, \
    SurroundInhibition, GaussianBlur
from STMD_cuda.utils.debug_show import tensor_show, tensor_save

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Lamina(nn.Module):
    def __init__(self):
        super(Lamina, self).__init__()
        self.hGammaBandPassFilter = GammaBandPassFilter()

    def init_config(self):
        self.hGammaBandPassFilter.init_config()

    def forward(self, laminaIpt):
        laminaOpt = self.hGammaBandPassFilter.process(laminaIpt)
        return laminaOpt


class Medulla(nn.Module):
    def __init__(self):
        super(Medulla, self).__init__()

        self.hPara5Mi1 = None
        self.hPara5Tm1 = None
        self.cellTm1Para3Ipt = None
        self.cellTm1Para5Ipt = None
        self.cellMi1Ipt = None
        self.create_flag = False

    def init_config(self):
        self.hTm1 = GammaDelay(5, 10)
        self.hMi1 = GammaDelay(5, 10)
        self.hPara5Mi1 = GammaDelay(25, 30)
        self.hPara5Tm1 = GammaDelay(25, 30)

        self.hTm1.init_config()
        self.hMi1.init_config()
        self.hPara5Mi1.init_config()
        self.hPara5Tm1.init_config()

    def forward(self, MedullaIpt):
        # Processing method
        # Applies processing to the input and returns the output
        # Process Tm2 and Tm3 components
        tm2Signal = torch.clamp(-MedullaIpt, min=0)
        tm3Signal = torch.clamp(MedullaIpt, min=0)

        tm1Para3Signal = self.hTm1.process(tm2Signal)
        mi1Para3Signal = self.hMi1.process(tm3Signal)
        tm1Para5Signal = self.hPara5Tm1.process(tm2Signal)
        mi1Para5Signal = self.hPara5Mi1.process(tm3Signal)
        # tensor_save('D:/CodeField/Bio-Inspired detection/research/STMD_cuda/MedullaImg/tm2.jpg', tm2Signal)
        # tensor_save('D:/CodeField/Bio-Inspired detection/research/STMD_cuda/MedullaImg/tm3.jpg', tm3Signal)
        # tensor_save('D:/CodeField/Bio-Inspired detection/research/STMD_cuda/MedullaImg/tm1.jpg', tm1Para5Signal)
        # tensor_save('D:/CodeField/Bio-Inspired detection/research/STMD_cuda/MedullaImg/Mi1.jpg', mi1Para5Signal)
        self.Opt = [tm3Signal, tm1Para3Signal, mi1Para5Signal, tm2Signal, mi1Para3Signal, tm1Para5Signal,
                    self.hPara5Mi1.tau]
        return self.Opt


class Lobula(nn.Module):
    def __init__(self):
        super(Lobula, self).__init__()
        self.hSTMD = None
        self.hLPTC = None

    def init_config(self):
        # Initialization method
        # This method initializes the Lobula layer component
        self.hSTMD = Stmdcell()
        self.hLPTC = Lptcell()
        self.hSTMD.init_config()
        self.hLPTC.init_config(self.hSTMD.hGammaDelay.lenKernel)

    def forward(self, varagein):
        # Processing method
        # Performs a correlation operation on the ON and OFF channels
        # and then applies surround inhibition

        # Extract ON and OFF channel signals from the input
        tm3Signal, tm1Para3Signal, mi1Para5Signal, tm2Signal, mi1Para3Signal, tm1Para5Signal, tau5 = varagein

        psi, fai = self.hLPTC.process(tm3Signal, mi1Para5Signal, tm2Signal, tm1Para5Signal, tau5)

        lobulaOpt = self.hSTMD.process(tm3Signal, tm1Para3Signal, psi, fai)

        self.Opt = lobulaOpt

        return lobulaOpt, []


class Stmdcell:
    def __init__(self):
        super(Stmdcell, self).__init__()
        self.hSubInhi = None
        self.alpha = 0.1
        self.sigma = 1.5
        self.gaussKernel = None  # Gaussian kernel
        self.hGammaDelay = None
        self.cellDPlusE = None
        self.paraGaussKernel = {'size': 12, 'eta': 1.5}
        self.create_flag = False
        self.gaussian_operator = GaussianBlur(kernel_size=self.paraGaussKernel['size'],
                                              sigma=self.paraGaussKernel['eta'])

        self.cellDPlusE_count = 0

    def init_config(self):
        # Initialization method
        # This method initializes the Lobula layer component
        self.hSubInhi = SurroundInhibition()
        self.hGammaDelay = GammaDelay(6, 12)

        self.hSubInhi.init_config()
        self.hGammaDelay.init_config()

    def tensor_renew(self, inputTensor):
        if self.cellDPlusE_count < self.cellDPlusE.shape[1]:
            self.cellDPlusE[0, self.cellDPlusE_count + 1:] = self.cellDPlusE[0,
                                                             self.cellDPlusE_count:-1].clone()
            self.cellDPlusE[0, 0:inputTensor.shape[1]] = inputTensor
            self.cellDPlusE_count += inputTensor.shape[1]
        else:
            self.cellDPlusE[0, 1:] = self.cellDPlusE[0, :-1].clone()
            self.cellDPlusE[0, 0:inputTensor.shape[1]] = inputTensor

    def process(self, tm3Signal, tm1Signal, faiList, psiList):
        # create cellDPlusE
        if self.create_flag is False:
            self.cellDPlusE = torch.zeros(
                (1, self.hGammaDelay.lenKernel, tm3Signal.shape[2], tm3Signal.shape[3])).to(device)
            self.create_flag = True

        # Processing method
        # Performs temporal convolution, correlation, and surround inhibition
        '''convIpt: (1, lenKernel, x, y)'''
        convIpt = torch.zeros_like(self.cellDPlusE)
        for idd in range(faiList.shape[0]):
            convIpt[:, idd, :, :] = translate_tensor(self.cellDPlusE[:, idd, :, :].clone().unsqueeze(0),
                                                     int(psiList[idd].item()), int(faiList[idd].item()))
        feedbackSignal = self.hGammaDelay.process(convIpt)

        # if torch.all(feedbackSignal != 0):
        #     feedbackSignal *= self.alpha
        #     correlationD = torch.clamp(tm3Signal - feedbackSignal, min=0) * torch.clamp(
        #         tm1Signal - feedbackSignal, min=0)
        # else:
        correlationD = torch.clamp(tm3Signal, min=0) * torch.clamp(tm1Signal, min=0)

        correlationE = self.gaussian_operator(tm3Signal * tm1Signal)

        # 侧抑制
        lateralInhiSTMDOpt = self.hSubInhi.process(correlationD)
        # 更新self.cellDPlusE
        self.tensor_renew(self.cellDPlusE)

        return lateralInhiSTMDOpt


class Lptcell:
    def __init__(self):
        super(Lptcell, self).__init__()
        # (n,1)
        self.betaTensor = torch.arange(2, 19, 2).unsqueeze(1)
        # (1,n)
        self.thetaTensor = torch.arange(0, 2 * torch.pi, torch.pi / 4).unsqueeze(0)
        self.lenBetaList = self.betaTensor.shape[0]
        self.lenThetaList = self.thetaTensor.shape[1]
        self.velocity = None
        self.sumV = None
        self.tuningCurvef = None
        self.shift_tensor = None

        '''生成平移矩阵'''
        # (9*8)
        self.shiftX = torch.round(self.betaTensor * torch.cos(self.thetaTensor + torch.pi / 2)).to(
            dtype=torch.int)
        self.shiftY = torch.round(self.betaTensor * torch.sin(self.thetaTensor + torch.pi / 2)).to(
            dtype=torch.int)

    def init_config(self, lenVelocity):
        self.velocity = torch.zeros(lenVelocity)
        self.sumV = torch.zeros(lenVelocity)
        # generate gauss distribution
        gaussianDistribution = torch.exp(-0.5 * (torch.range(-200, 199) / (100 / 2)) ** 2)
        # 归一化
        gaussianDistribution /= torch.max(gaussianDistribution)

        """生成self.tuningCurvef"""
        self.tuningCurvef = torch.zeros((self.lenBetaList, self.lenThetaList * 100 + 200))
        self.tuningCurvef[0, :300] = gaussianDistribution[100:400]
        self.tuningCurvef[-1, -300:] = gaussianDistribution[:300]
        for id in range(1, self.lenBetaList - 1):
            idRange = slice((id + 1) * 100 - 200, (id + 1) * 100 + 200)
            self.tuningCurvef[id, idRange] = gaussianDistribution

    def process(self, tm1Signal, tm2Signal, tm3Signal, mi1Signal, tau5):
        """
        tm1Signal, tm2Signal, tm3Signal, mi1Signal, (1,1,height,width)
        """
        sumLptcOptR = torch.zeros((self.lenBetaList, self.lenThetaList))

        for idBeta in range(self.lenBetaList):
            for idTheta in range(self.lenThetaList):
                shiftMi1Signal = translate_tensor(mi1Signal.clone(),
                                                  self.shiftX[idBeta, idTheta].item(),
                                                  self.shiftY[idBeta, idTheta].item())
                shiftTm1Signal = translate_tensor(tm1Signal.clone(),
                                                  self.shiftX[idBeta, idTheta].item(),
                                                  self.shiftY[idBeta, idTheta].item())

                ltpcOpt = tm3Signal * shiftMi1Signal + tm2Signal * shiftTm1Signal
                # tensor_show('lptcOpt', ltpcOpt)
                sumLptcOptR[idBeta, idTheta] = torch.sum(ltpcOpt)

        # preferTheta
        '''获取最大值索引'''
        max_value_index = torch.argmax(sumLptcOptR)
        max_row, max_col = divmod(max_value_index.item(), sumLptcOptR.size(1))
        maxTheta = self.thetaTensor[0, max_col]

        # background velocity
        self.velocity = torch.roll(self.velocity, shifts=-1, dims=0)
        firingRate = sumLptcOptR[:, max_col]

        '''
        firingRate.unsqueeze(1) (9,1)
        self.tuningCurvef (9,y)
        broadcasting
        '''
        temp = torch.exp(-(firingRate.unsqueeze(1) - self.tuningCurvef) ** 2)
        temp = torch.prod(temp, dim=0)
        self.velocity[-1] = torch.argmax(temp)

        for idV in range(self.velocity.shape[0] - 1, -1, -1):
            self.sumV[idV] = torch.sum(self.velocity[idV:])

        fai = self.sumV * torch.cos(maxTheta)
        psi = self.sumV * torch.sin(maxTheta)

        return fai, psi


def translate_tensor(input_tensor, shiftX, shiftY):
    """
    input_tensor: gpu
    shift_x: number
    shift_y: number
    """
    # init output
    # translated_tensor = torch.zeros_like(input_tensor)
    m, n = input_tensor.shape[2], input_tensor.shape[3]
    if abs(shiftX) >= n or abs(shiftY) >= m:
        return torch.zeros_like(input_tensor)

    # 平移
    input_tensor = torch.roll(input_tensor, shifts=(
        shiftX, shiftY), dims=(3, 2))

    # 填充0
    # down
    if shiftY > 0:
        input_tensor[:, :, :shiftY, :] = 0
    # up
    elif shiftY < 0:
        input_tensor[:, :, shiftY:, :] = 0
    # right
    if shiftX > 0:
        input_tensor[:, :, :, :shiftX] = 0
    # left
    elif shiftX < 0:
        input_tensor[:, :, :, shiftX:] = 0

    return input_tensor
