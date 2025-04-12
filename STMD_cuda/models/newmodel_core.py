import os.path

from STMD_cuda.models.math_operator import GaussianBlur
from torch import nn
from torch.nn import functional as F
from STMD_cuda.utils.debug_show import tensor_show, tensor_save, tensor_save_txt_row, ndarray_save
import torch
import numpy as np
from STMD_cuda.data.dataloader import VidstreamReader, ImgstreamReader, Visualization
from STMD_cuda.models.math_operator import SurroundInhibition
from STMD_cuda.models import backbonev2_core
from STMD_cuda.models import YTSmodel_core

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Retina(nn.Module):
    def __init__(self):
        super(Retina, self).__init__()
        self.pre_retinaIpt = None
        self.diff_retinaIpt = None

    def init_config(self):
        pass

    def forward(self, retinaIpt):
        if self.pre_retinaIpt is None:
            self.pre_retinaIpt = torch.zeros_like(retinaIpt).to(device)

        self.diff_retinaIpt = retinaIpt - self.pre_retinaIpt
        self.pre_retinaIpt = retinaIpt
        ''''''
        tensor_show('diff_retina', self.diff_retinaIpt)
        ''''''
        return self.diff_retinaIpt


class Lamina(nn.Module):
    def __init__(self):
        super(Lamina, self).__init__()
        self.hGaussianFilter1 = GaussianBlur(kernel_size=5, sigma=2.0)
        self.hGaussianFilter2 = GaussianBlur(kernel_size=9, sigma=4.0)
        self.preL1 = None
        self.preL2 = None
        self.taui = 1000 / 25
        self.tau1 = 1.0
        self.tau2 = 100.0
        self.alpha1 = self.taui / (self.tau1 + self.taui)
        self.alpha2 = self.taui / (self.tau2 + self.taui)

    def init_config(self):
        pass

    def forward(self, laminaIpt):
        Pe = self.hGaussianFilter1(laminaIpt)
        Pi = self.hGaussianFilter2(laminaIpt)
        ''''''
        tensor_show('pe', Pe)
        tensor_show('pi', Pi)
        ''''''
        judge = torch.zeros_like(Pe).to(device)
        judge[(Pe >= 0) & (Pi >= 0)] = 1
        judge[(Pe < 0) & (Pi < 0)] = -1
        LA = torch.abs(Pe - Pi)
        LA *= judge
        ''''''
        tensor_show('LA', LA)
        ''''''
        L1 = torch.clamp(LA, min=0)
        L2 = torch.clamp(-LA, min=0)

        if self.preL1 is None:
            self.preL1 = torch.zeros_like(L1).to(device)
            self.preL2 = torch.zeros_like(L2).to(device)

        delta_L1 = L1 - self.preL1
        delta_L2 = L2 - self.preL2

        judge_part1 = torch.zeros_like(delta_L1).to(device)
        judge_part1[delta_L1 >= 0] = self.alpha1
        judge_part1[delta_L1 < 0] = self.alpha2

        L1_star = judge_part1 * L1 + (1 - judge_part1) * self.preL1

        judge_part2 = torch.zeros_like(delta_L2).to(device)
        judge_part2[delta_L2 >= 0] = self.alpha1
        judge_part2[delta_L2 < 0] = self.alpha2

        L2_star = judge_part2 * L2 + (1 - judge_part2) * self.preL2

        ''''''
        tensor_show('L1star', L1_star)
        tensor_show('L2star', L2_star)
        ''''''

        M1 = L1 - L1_star
        M2 = L2 - L2_star

        self.preL1 = L1
        self.preL2 = L2
        ''''''
        tensor_show('M1', M1)
        tensor_show('M2', M2)
        ''''''
        return [M1, M2]


class Medulla(nn.Module):
    def __init__(self):
        super(Medulla, self).__init__()
        self.preOnSignal = None
        self.preOffSignal = None
        self.alpha = 0.5

    def init_config(self):
        pass

    def forward(self, medullaIpt):
        onSignal = medullaIpt[0]
        offSignal = medullaIpt[1]

        if self.preOnSignal is None:
            self.preOnSignal = 0
            self.preOffSignal = 0
            onSignal_delay = onSignal
            offSignal_delay = offSignal
        else:
            onSignal_delay = self.alpha * (onSignal + self.preOnSignal)
            offSignal_delay = self.alpha * (offSignal + self.preOffSignal)

        self.preOnSignal = onSignal
        self.preOffSignal = offSignal

        return [onSignal, offSignal, onSignal_delay, offSignal_delay]


class Lobula(nn.Module):
    def __init__(self):
        super(Lobula, self).__init__()
        self.hSubInhi = SurroundInhibition()

    def init_config(self):
        self.hSubInhi.init_config()

    def forward(self, lobulaIpt):
        lobulaOpt = lobulaIpt[0] * lobulaIpt[3] + lobulaIpt[1] * lobulaIpt[2]
        lobulaOpt = self.hSubInhi.process(lobulaOpt)
        return lobulaOpt


class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        self.hRetina = backbonev2_core.Retina()
        self.hLamina = backbonev2_core.Lamina()
        self.hMedulla = YTSmodel_core.Medulla()
        self.hLobula = YTSmodel_core.Lobula()
        self.modelOpt = {}

        self.tm3_path = 'D:/CodeField/Bio-Inspired detection/research/Tm3'
        self.tm1_path = 'D:/CodeField/Bio-Inspired detection/research/Tm1'
        self.stmd_path = 'D:/CodeField/Bio-Inspired detection/research/STMD_Img'
        self.detresult_path = 'D:/CodeField/Bio-Inspired detection/research/Detect_Result'
        self.tm2_path = 'D:/CodeField/Bio-Inspired detection/research/Tm2'
        self.mi1_path = 'D:/CodeField/Bio-Inspired detection/research/Mi1'
        self.lmc_path = 'D:/CodeField/Bio-Inspired detection/research/laminaOpt'
        self.counter = 0

    def init_config(self):
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()

    def forward(self, iptTensor):
        self.counter += 1
        self.retinaOpt = self.hRetina(iptTensor)
        # tensor_show("retinaopt", self.retinaOpt)
        self.laminaOpt = self.hLamina(self.retinaOpt)
        tensor_save(os.path.join(self.lmc_path, str(self.counter) + 'as.jpg'), self.laminaOpt)
        tensor_save_txt_row(os.path.join(self.lmc_path, str(self.counter) + 'as.txt'), self.laminaOpt)
        # tensor_show("laminaopt", self.laminaOpt)
        self.medullaOpt = self.hMedulla(self.laminaOpt)
        # tensor_save(os.path.join(self.tm3_path, str(self.counter) + 'as.jpg'), self.medullaOpt[0])
        # tensor_save(os.path.join(self.tm1_path, str(self.counter) + 'as.jpg'), self.medullaOpt[4])
        # tensor_save(os.path.join(self.tm2_path, str(self.counter) + 'as.jpg'), self.medullaOpt[1])
        # tensor_save(os.path.join(self.mi1_path, str(self.counter) + 'as.jpg'), self.medullaOpt[5])
        # tensor_save_txt_row(os.path.join(self.tm3_path, str(self.counter) + 'as.txt'), self.medullaOpt[0])
        # tensor_save_txt_row(os.path.join(self.tm1_path, str(self.counter) + 'as.txt'), self.medullaOpt[4])
        # tensor_save_txt_row(os.path.join(self.tm2_path, str(self.counter) + 'as.txt'), self.medullaOpt[1])
        # tensor_save_txt_row(os.path.join(self.mi1_path, str(self.counter) + 'as.txt'), self.medullaOpt[5])
        self.lobulaOpt = self.hLobula(self.medullaOpt)
        # tensor_save(os.path.join(self.stmd_path, str(self.counter) + 'as.jpg'), self.lobulaOpt['response'])
        # tensor_save_txt_row(os.path.join(self.stmd_path, str(self.counter) + 'as.txt'), self.lobulaOpt['response'])
        return self.lobulaOpt


import glob


def detector():
    objIptStream = ImgstreamReader(startImgName='D:/CodeField/Bio-Inspired detection/Dataset_linshi/v1frame/frame_0020.jpg', endImgName='D:/CodeField/Bio-Inspired detection/Dataset_linshi/v1frame/frame_0060.jpg')
    objIptStream_test = ImgstreamReader(startImgName='D:/CodeField/Bio-Inspired detection/Dataset_linshi/v1frame/frame_0020.jpg', endImgName='D:/CodeField/Bio-Inspired detection/Dataset_linshi/v1frame/frame_0060.jpg')
    ojbVisualize = Visualization()
    objModel = NewModel()
    objModel.init_config()
    objModel.hLobula.n = 'tau3'
    objModel.hLobula.k = 0
    objModel.hMedulla.n = 'tau3'

    counter = 0
    while objIptStream.hasFrame:
        print(counter)
        grayImg, colorImg = objIptStream.get_next_frame()
        colorImg1 = colorImg
        if counter >= 1:
            grayImg1, colorImg1 = objIptStream_test.get_next_frame()
        iptTensor = torch.tensor(grayImg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        result = objModel.forward(iptTensor)
        # pre_colorImg = colorImg
        # if counter <= 1:
        #     counter += 1
        #     continue
        frame_data = ojbVisualize.show_result(colorImg1, result)
        counter += 1


if __name__ == '__main__':
    detector()

# import glob

# if __name__ == '__main__':
#
#     # folder_path = "F:/Documents/selfbuiltDataset/simulatedDataset/detection_lag"
#     folder_path = 'D:/CodeField/Bio-Inspired detection/Dataset_linshi'
#
#     # 读取文件夹内所有视频文件并排序
#     # 支持的视频文件扩展名列表
#     video_extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.flv', '*.wmv']
#
#     # 初始化一个空列表来存储所有找到的视频文件
#     video_files = []
#
#     # 遍历所有扩展名并查找匹配的文件
#     for ext in video_extensions:
#         video_files.extend(glob.glob(os.path.join(folder_path, ext)))
#
#     # 按照文件名排序
#     video_files.sort()
#     video_counter = 0
#     # 对所有视频进行循环
#     for idVideo in video_files:
#         print("---------video ", video_counter, "-----------\n")
#         print(idVideo)
#         objIptStream = VidstreamReader(vidName=idVideo)
#         ojbVisualize = Visualization()
#         objModel = NewModel()
#         objModel.init_config()
#
#         counter = 0
#         step = 1
#         pre_colorImg = None
#         n3tau3_list = ['tau3', 'tau6', 'tau9', 'tau12', 'tau15', 'tau18', 'tau21', 'tau24']
#         n4tau4_list = ['tau3', 'tau4', 'tau5', 'tau6', 'tau7']
#         k_list = [0, 0.01, 0.1, 0.25, 0.5, 1]
#
#         inner_data = ''
#         # 对所有k值进行循环
#         for k in n3tau3_list:
#             # 更换model中的k值
#             objModel.hLobula.n = n4tau4_list[0]
#             objModel.hLobula.k = k_list[0]
#
#             objModel.hMedulla.n = k
#
#             while objIptStream.hasFrame:
#                 print(counter)
#                 grayImg, colorImg = objIptStream.get_next_frame()
#                 # ndarray_save(os.path.join('D:/CodeField/Bio-Inspired detection/research/ColorImg', str(counter + 1) + '.jpg'),
#                 #              colorImg)
#                 # print(counter)
#                 iptTensor = torch.tensor(grayImg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
#                 result = objModel.forward(iptTensor)
#                 # pre_colorImg = colorImg
#                 # if counter <= 1:
#                 #     counter += 1
#                 #     continue
#                 frame_data = ojbVisualize.show_result(colorImg, result)
#                 if counter == 10:
#                     # print("k: ", k, "Model Output value: ", objModel.hLobula.max_val.item())
#                     print("k: ", k, "Model Output value: ", objModel.hLobula.hLPTC.lptc_test)
#                     for idx in objModel.hLobula.hLPTC.lptc_test:
#                         inner_data += str(idx) + ' '
#                     counter = 0
#                     objModel.hLobula.hLPTC.lptc_test = []
#                     break
#
#                 objModel.hLobula.hLPTC.lptc_test = []
#                 counter += 1
#             # print("k: ", k, "Model Output value: ",
#             #       sum(objModel.hLobula.maxVal_list) / len(objModel.hLobula.maxVal_list), "\n")
#             print("--------------------------next epoch----------------------------\n")
#             # 清空视频读取痕迹，重新开始下一轮
#             objIptStream.hasFrame = objIptStream.hVid.isOpened()
#             objIptStream.currIdx = 0
#             objIptStream.frameIdx = 0
#             counter = 0
#         video_counter += 1
#         print("--------------------------next video----------------------------\n")
