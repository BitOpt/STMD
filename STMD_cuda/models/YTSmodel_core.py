import os.path
import time

import cv2
import torch
from torch import nn
from STMD_cuda.utils.debug_show import tensor_save_txt, tensor_show, tensor_save, lptc_save
from STMD_cuda.models.math_operator import GaussianBlur, SurroundInhibition
from torch.nn import functional as F
import numpy as np
from sklearn.cluster import DBSCAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Retina(nn.Module):
    def __init__(self):
        super(Retina, self).__init__()
        self.hGaussianBlur = GaussianBlur(kernel_size=3, sigma=1.0)

    def init_config(self):
        pass

    def forward(self, iptTensor):
        retinaOpt = self.hGaussianBlur(iptTensor)
        return retinaOpt


class Medulla(nn.Module):
    def __init__(self):
        super(Medulla, self).__init__()
        self.cellGammaTm2 = None
        self.cellGammaTm3 = None
        self.lenCell = 10
        self.create_flag = False
        self.cellCount = 0

        self.n = 3
        # self.tau = 20

        self.weight = {'tau3': torch.tensor([0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]).view(1, -1, 1, 1).to(device),
                       'tau6': torch.tensor([0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0]).view(1, -1, 1, 1).to(device),
                       'tau9': torch.tensor([0, 0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0]).view(1, -1, 1, 1).to(device),
                       'tau12': torch.tensor([0, 0, 0.1, 0.8, 0.1, 0, 0, 0, 0, 0]).view(1, -1, 1, 1).to(device),
                       'tau15': torch.tensor([0, 0, 0, 0.1, 0.8, 0.1, 0, 0, 0, 0]).view(1, -1, 1, 1).to(device),
                       'tau18': torch.tensor([0, 0, 0, 0, 0.1, 0.8, 0.1, 0, 0, 0]).view(1, -1, 1, 1).to(device),
                       'tau21': torch.tensor([0, 0, 0, 0, 0, 0.1, 0.8, 0.1, 0, 0]).view(1, -1, 1, 1).to(device),
                       'tau24': torch.tensor([0, 0, 0, 0, 0, 0, 0.1, 0.8, 0.1, 0]).view(1, -1, 1, 1).to(device)}

    def init_config(self):
        pass

    def forward(self, MedullaIpt):
        tm2Signal = torch.clamp(-MedullaIpt, min=0)
        tm3Signal = torch.clamp(MedullaIpt, min=0)

        if self.create_flag is False:
            self.cellGammaTm2 = torch.zeros(1, self.lenCell, tm2Signal.shape[2], tm2Signal.shape[3]).to(device)
            self.cellGammaTm3 = torch.zeros(1, self.lenCell, tm3Signal.shape[2], tm3Signal.shape[3]).to(device)
            self.create_flag = True

        tm1Para5Signal = None
        Mi1Para5Signal = None

        tm1Para3Signal = torch.sum(self.cellGammaTm2 * self.weight[self.n], dim=1, keepdim=True)
        Mi1Para3Signal = torch.sum(self.cellGammaTm3 * self.weight[self.n], dim=1, keepdim=True)

        self.cellGammaTm2[0, tm2Signal.shape[1]:] = self.cellGammaTm2[0, 0:self.lenCell - tm2Signal.shape[1]].clone()
        self.cellGammaTm2[0, 0:tm2Signal.shape[1]] = tm2Signal
        self.cellGammaTm3[0, tm3Signal.shape[1]:] = self.cellGammaTm3[0, 0:self.lenCell - tm3Signal.shape[1]].clone()
        self.cellGammaTm3[0, 0:tm3Signal.shape[1]] = tm3Signal

        return [tm3Signal, tm2Signal, Mi1Para5Signal, tm1Para5Signal, tm1Para3Signal, Mi1Para3Signal]


class Lobula(nn.Module):
    def __init__(self):
        super(Lobula, self).__init__()
        self.hLPTC = None
        self.preLPTC_on = None
        self.preLPTC_off = None
        self.feedback_on = None
        self.feedback_off = None
        self.cell_lptc_on = None
        self.cell_lptc_off = None
        self.k = 0.5  # 0.01
        self.n = 3
        self.maxVal_list = []
        self.max_val = 0

        self.weight = {'tau3': torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).view(1, -1, 1, 1).to(device),
                       'tau4': torch.tensor([0.8, 0.2, 0, 0, 0, 0, 0, 0, 0, 0]).view(1, -1, 1, 1).to(device),
                       'tau5': torch.tensor([0.6, 0.4, 0, 0, 0, 0, 0, 0, 0, 0]).view(1, -1, 1, 1).to(device),
                       'tau6': torch.tensor([0.4, 0.6, 0, 0, 0, 0, 0, 0, 0, 0]).view(1, -1, 1, 1).to(device),
                       'tau7': torch.tensor([0.2, 0.8, 0, 0, 0, 0, 0, 0, 0, 0]).view(1, -1, 1, 1).to(device)}
        self.lenCell = 10
        self.create_flag = False

    def init_config(self):
        self.hLPTC = Lptcell()
        self.hLPTC.init_config()

    def forward(self, LobulaIpt):
        tm3Signal, tm2Signal, Mi1Para5Signal, tm1Para5Signal, tm1Para3Signal, Mi1Para3Signal = LobulaIpt

        if self.create_flag is False:
            self.cell_lptc_on = torch.zeros(1, self.lenCell, tm3Signal.shape[2], tm3Signal.shape[3]).to(device)
            self.cell_lptc_off = torch.zeros(1, self.lenCell, tm3Signal.shape[2], tm3Signal.shape[3]).to(device)
            self.feedback_on = torch.zeros(1, 1, tm3Signal.shape[2], tm3Signal.shape[3]).to(device)
            self.feedback_off = torch.zeros(1, 1, tm2Signal.shape[2], tm2Signal.shape[3]).to(device)
            self.create_flag = True

        self.feedback_on = torch.sum(self.cell_lptc_on * self.weight[self.n], dim=1, keepdim=True)
        self.feedback_off = torch.sum(self.cell_lptc_off * self.weight[self.n], dim=1, keepdim=True)
        LPTC_on = (tm3Signal + self.k * self.feedback_on) * (tm1Para3Signal + self.k * self.feedback_on)
        LPTC_off = (tm2Signal + self.k * self.feedback_off) * (Mi1Para3Signal + self.k * self.feedback_off)

        motion_target = LPTC_on + LPTC_off

        self.cell_lptc_on[0, LPTC_on.shape[1]:] = self.cell_lptc_on[0, 0:self.lenCell - LPTC_on.shape[1]].clone()
        self.cell_lptc_on[0, 0:LPTC_on.shape[1]] = LPTC_on
        self.cell_lptc_off[0, LPTC_off.shape[1]:] = self.cell_lptc_off[0, 0:self.lenCell - LPTC_off.shape[1]].clone()
        self.cell_lptc_off[0, 0:LPTC_off.shape[1]] = LPTC_off

        # tensor_show("target", motion_target)
        squeezed = motion_target.squeeze()

        # 求张量中最大值
        self.max_val = torch.max(squeezed)
        if self.max_val.item() >= 0:
            self.maxVal_list.append(self.max_val.item())

        # 求张量中所有元素的和
        # self.maxVal_list.append(torch.sum(squeezed[squeezed > 0]).item())

        # sumLptcOptR = self.hLPTC.process(tm3Signal, tm2Signal, Mi1Para3Signal, tm1Para3Signal)
        # if len(sumLptcOptR) == 0:   sumLptcOptR = None
        sumLptcOptR = None
        return {'response': motion_target, 'direction': sumLptcOptR}


class Lptcell:
    def __init__(self):
        super(Lptcell, self).__init__()
        # (n,1)
        self.betaTensor = torch.arange(2, 19, 4).unsqueeze(1)
        # (1,n)
        self.thetaTensor = torch.arange(0, 1, 2).unsqueeze(0)
        self.lenBetaList = self.betaTensor.shape[0]
        self.lenThetaList = self.thetaTensor.shape[1]

        self.shiftX = torch.round(self.betaTensor * torch.cos(self.thetaTensor)).to(
            dtype=torch.int)
        self.shiftY = torch.round(self.betaTensor * torch.sin(self.thetaTensor)).to(
            dtype=torch.int)
        self.LPTCsave_path = 'D:/CodeField/Bio-Inspired detection/research/LPTCimg'
        self.shiftMisave_path = 'D:/CodeField/Bio-Inspired detection/research/shiftMi1'
        self.labeldetect_path = 'D:/CodeField/Bio-Inspired detection/research/labeldetect'

        self.lptc_test = []

    def init_config(self):
        pass

    def process(self, tm3Signal, tm2Signal, Mi1Signal, tm1Signal):
        sumLptcOptR = torch.zeros((self.lenBetaList, self.lenThetaList))
        shiftMi1SignalSUM = torch.zeros_like(Mi1Signal)
        shiftTm1SignalSUM = torch.zeros_like(tm1Signal)
        lptcSUM = torch.zeros_like(tm1Signal)
        lptcList = []

        tm3_ndarray = tm3Signal.clone().squeeze().cpu().numpy()
        min = tm3_ndarray.min()
        max = tm3_ndarray.max()
        plot_nor = (tm3_ndarray - min) / (max - min) * 255
        tm3_ndarray = cv2.applyColorMap(plot_nor.astype(np.uint8), cv2.COLORMAP_JET)

        lptc_coordinates = []
        lptc_values = []

        counter = 1

        with open("D:/CodeField/Bio-Inspired detection/research/lptccoor.txt", "w") as file:
            for idBeta in range(self.lenBetaList):
                for idTheta in range(self.lenThetaList):
                    shiftMi1Signal = translate_tensor(Mi1Signal.clone(), self.shiftX[idBeta, idTheta].item(),
                                                      self.shiftY[idBeta, idTheta].item())
                    # shiftTm1Signal = translate_tensor(tm1Signal.clone(), self.shiftX[idBeta, idTheta].item(),
                    #                                   self.shiftY[idBeta, idTheta].item())
                    lptcOpt = tm3Signal * shiftMi1Signal  # + tm2Signal * shiftTm1Signal
                    # file.write(f'{self.betaTensor[idBeta].item()} {self.theta[idTheta]} {lptcOpt.sum().item()}\n')
                    lptcOpt[lptcOpt < 1e-3] = 0
                    # tensor_save(os.path.join(self.shiftMisave_path, str(counter) + '.jpg'), shiftMi1Signal)
                    # tensor_save(os.path.join(self.LPTCsave_path, str(counter) + '.jpg'), lptcOpt)
                    counter += 1
                    # tensor_show('tm3shiftmi1', tm3Signal + shiftMi1Signal)
                    # tensor_show('lptcOpt', lptcOpt)
                    # sumLptcOptR[idBeta, idTheta] = torch.sum(lptcOpt)
                    shiftMi1SignalSUM += shiftMi1Signal
                    # shiftTm1SignalSUM += shiftTm1Signal
                    lptcSUM += lptcOpt
                    # lptcList.append(lptcOpt)

                    # 找到极大值及其坐标位置
                    chazhi1 = torch.where(lptcOpt - translate_tensor(lptcOpt, 1, 0) <= 0,
                                          torch.tensor(-1.0, device='cuda'),
                                          torch.tensor(0.0, device='cuda'))
                    chazhi2 = torch.where(lptcOpt - translate_tensor(lptcOpt, 0, 1) <= 0,
                                          torch.tensor(-1.0, device='cuda'),
                                          torch.tensor(0.0, device='cuda'))
                    chazhi3 = torch.where(lptcOpt - translate_tensor(lptcOpt, -1, 0) <= 0,
                                          torch.tensor(-1.0, device='cuda'),
                                          torch.tensor(0.0, device='cuda'))
                    chazhi4 = torch.where(lptcOpt - translate_tensor(lptcOpt, 0, -1) <= 0,
                                          torch.tensor(-1.0, device='cuda'),
                                          torch.tensor(0.0, device='cuda'))
                    chazhi_round = chazhi1 + chazhi2 + chazhi3 + chazhi4
                    zeros_coords = (chazhi_round == 0).nonzero(as_tuple=False).cpu().numpy()

                    # 需要加入判断，如果无小目标直接跳出此次循环
                    if zeros_coords.size == 0:
                        self.lptc_test.append(0)
                        continue

                    height_indices = zeros_coords[:, 2]
                    width_indices = zeros_coords[:, 3]
                    values = lptcOpt[0, 0, height_indices, width_indices].cpu().numpy()

                    self.lptc_test.append(np.max(values))

                    zeros_coords_with_bias = np.hstack((zeros_coords, np.tile(
                        np.array([self.shiftX[idBeta, idTheta].item(), self.shiftY[idBeta, idTheta].item()]),
                        (zeros_coords.shape[0], 1))))
                    lptc_coordinates.extend(list(zeros_coords_with_bias))
                    lptc_values.extend(list(values))

        lptcOutput = []
        if len(lptc_coordinates) != 0:
            # 大分辨率场景下，获得的lptc点太多导致计算速度超级慢，因此设置阈值减少lptc点，根据lptc_values取前50个点
            max50_indices = np.argsort(np.array(lptc_values))[-50:]
            lptc_coordinates = np.array(lptc_coordinates)[max50_indices]
            lptc_values = np.array(lptc_values)[max50_indices]

            #######
            coordinates_path = os.path.join(self.LPTCsave_path, 'coordinates.txt')
            lptc_values_expand = lptc_values[:, np.newaxis]
            combined = np.hstack((lptc_coordinates[:, 2:4], lptc_values_expand))
            tensor_save_txt(coordinates_path, combined)
            #######

            dbscan = DBSCAN(eps=10, min_samples=1)
            labels = dbscan.fit_predict(lptc_coordinates[:, 2:4])
            group_coordinates = {}
            for i in range(lptc_coordinates.shape[0]):
                label = labels[i]
                if label not in group_coordinates:
                    group_coordinates[label] = []
                group_coordinates[label].append((lptc_coordinates[:, 2:4][i], np.array(lptc_values)[i]))
            for label, coordinates in group_coordinates.items():
                coords_array = np.array(coordinates)
                max_index = np.argmax(coords_array[:, 1].astype(np.float32))
                max_coordinate = coords_array[max_index]
                print(f"label {label}: max coordinate is {max_coordinate[0]}, value is {max_coordinate[1]}")
                rows = np.where((lptc_coordinates[:, 2] == max_coordinate[0][0]) & (
                        lptc_coordinates[:, 3] == max_coordinate[0][1]))[0]
                lptcOutput.append(lptc_coordinates[rows[0]])

        # shiftMi1SignalSUM += Mi1Signal
        # shiftTm1SignalSUM += tm1Signal
        # for i, element in enumerate(lptcList):
        #     img_name = str(i) + '.jpg'
        #     img_path = os.path.join(self.LPTCsave_path, img_name)
        #     tensor_save(img_path, lptcList[i])

        # tensor_show('tm3Signal', tm3Signal)
        # tensor_show('tm2Signal', tm2Signal)
        # tensor_save(os.path.join(self.LPTCsave_path, 'shiftmi1sum.jpg'), shiftMi1SignalSUM)
        # tensor_show("shiftTm1SignalSUM", shiftTm1SignalSUM)
        # tensor_show('tm3plusmi1', tm3Signal + Mi1Signal)
        # tensor_save(os.path.join(self.LPTCsave_path, 'lptcsum.jpg'), lptcSUM)
        # cv2.imshow("tm3location", tm3_ndarray)
        # cv2.waitKey(1)
        # print("over")
        return lptcOutput


def translate_tensor(input_tensor, shiftX, shiftY):
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
