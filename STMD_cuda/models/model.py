import os.path

import torch
from torch import nn
from STMD_cuda.models import backbonev2_core, stfeedbackstmd_core
import numpy as np
from STMD_cuda.utils.debug_show import ndarray_show, tensor_show, tensor_save_txt, tensor_save_txt_row, tensor_save


class Backbonev2(nn.Module):
    def __init__(self):
        super(Backbonev2, self).__init__()
        self.hRetina = backbonev2_core.Retina()
        self.hLamina = backbonev2_core.Lamina()
        self.hMedulla = backbonev2_core.Medulla()
        self.hLobula = backbonev2_core.Lobula()

        self.hLamina.alpha = 0.3

        self.tm1_path = 'D:/CodeField/Bio-Inspired detection/research/Tm1'
        self.tm3_path = 'D:/CodeField/Bio-Inspired detection/research/Tm3'
        self.tm2_path = 'D:/CodeField/Bio-Inspired detection/research/Tm2'
        self.mi1_path = 'D:/CodeField/Bio-Inspired detection/research/Mi1'
        self.stmdimg_path = 'D:/CodeField/Bio-Inspired detection/research/STMD_Img'
        self.detectresult_path = 'D:/CodeField/Bio-Inspired detection/research/Detect_Result'
        self.laminaOpt_path = 'D:/CodeField/Bio-Inspired detection/research/laminaOpt'
        self.counter = 1

    def init_config(self):
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()

    def forward(self, modelIpt):
        retinaOpt = self.hRetina(modelIpt)
        laminaOpt = self.hLamina(retinaOpt)
        tensor_save(os.path.join(self.laminaOpt_path, str(self.counter)+'as.jpg'), laminaOpt)
        tensor_save_txt_row(os.path.join(self.laminaOpt_path, str(self.counter)+'as.txt'), laminaOpt)
        max_value = torch.max(laminaOpt)
        min_value = torch.min(laminaOpt)
        laminaOpt_cor = laminaOpt.clone()
        diff = max_value - min_value
        if diff >= 0.1:
            laminaOpt_cor[
                (0.2 * min_value <= laminaOpt_cor) & (laminaOpt_cor <= 0.2 * max_value)] = 0
        else:
            laminaOpt_cor.fill_(0)

        self.hMedulla(laminaOpt)
        medullaOpt = self.hMedulla.Opt
        # tensor_save_txt(os.path.join(self.tm3_path, str(self.counter)+'.txt'), medullaOpt[0])
        # tensor_save_txt(os.path.join(self.tm1_path, str(self.counter)+'.txt'), medullaOpt[1])

        # tensor_save(os.path.join(self.tm3_path, str(self.counter)+'as.jpg'), medullaOpt[0])
        # tensor_save(os.path.join(self.tm1_path, str(self.counter) + 'as.jpg'), medullaOpt[1])
        # tensor_save_txt_row(os.path.join(self.tm3_path, str(self.counter)+'as.txt'), medullaOpt[0])
        # tensor_save_txt_row(os.path.join(self.tm1_path, str(self.counter)+'as.txt'), medullaOpt[1])
        lobulaOpt, direction, _ = self.hLobula(medullaOpt[0], medullaOpt[1], laminaOpt_cor)
        # tensor_save(os.path.join(self.stmdimg_path, str(self.counter) + 'as.jpg'), lobulaOpt)
        # tensor_save_txt_row(os.path.join(self.stmdimg_path, str(self.counter)+'as.txt'), lobulaOpt)

        modelOpt = {}
        modelOpt['response'] = lobulaOpt
        modelOpt['direction'] = direction
        self.counter += 1

        return modelOpt


class STFeedbackSTMD(nn.Module):
    def __init__(self):
        super(STFeedbackSTMD, self).__init__()
        self.hRetina = backbonev2_core.Retina()
        self.hLamina = stfeedbackstmd_core.Lamina()
        self.hMedulla = stfeedbackstmd_core.Medulla()
        self.hLobula = stfeedbackstmd_core.Lobula()

        self.tm1_path = 'D:/CodeField/Bio-Inspired detection/research/Tm1'
        self.tm3_path = 'D:/CodeField/Bio-Inspired detection/research/Tm3'
        self.stmdimg_path = 'D:/CodeField/Bio-Inspired detection/research/STMD_Img'
        self.detectresult_path = 'D:/CodeField/Bio-Inspired detection/research/Detect_Result'
        self.laminaOpt_path = 'D:/CodeField/Bio-Inspired detection/research/laminaOpt'
        self.counter = 1

    def init_config(self):
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()

    def forward(self, iptTensor):
        self.retinaOpt = self.hRetina(iptTensor)
        self.laminaOpt = self.hLamina(self.retinaOpt)
        # tensor_show('laminaOpt', self.laminaOpt)
        max_value = torch.max(self.laminaOpt)
        min_value = torch.min(self.laminaOpt)
        self.laminaOpt_cor = self.laminaOpt.clone()
        diff = max_value - min_value
        if diff >= 0.1:
            self.laminaOpt_cor[
                (0.2 * min_value <= self.laminaOpt_cor) & (
                        self.laminaOpt_cor <= 0.2 * max_value)] = 0
        else:
            self.laminaOpt_cor.fill_(0)

        self.medullaOpt = self.hMedulla(self.laminaOpt_cor)
        # tensor_save(os.path.join(self.tm3_path, str(self.counter)+'as.jpg'), self.medullaOpt[0])
        # tensor_save(os.path.join(self.tm1_path, str(self.counter) + 'as.jpg'), self.medullaOpt[1])
        # tensor_save_txt_row(os.path.join(self.tm3_path, str(self.counter)+'as.txt'), self.medullaOpt[0])
        # tensor_save_txt_row(os.path.join(self.tm1_path, str(self.counter)+'as.txt'), self.medullaOpt[1])
        self.lobulaOpt, _ = self.hLobula(self.medullaOpt)
        # tensor_save(os.path.join(self.stmdimg_path, str(self.counter) + 'as.jpg'), self.lobulaOpt)
        # tensor_save_txt_row(os.path.join(self.stmdimg_path, str(self.counter)+'as.txt'), self.lobulaOpt)
        self.modelOpt = {}
        self.modelOpt['response'] = self.lobulaOpt
        self.modelOpt['direction'] = None

        self.counter += 1
        return self.modelOpt
