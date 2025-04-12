import os
import sys
import cv2

from STMD_cuda.data.dataloader import VidstreamReader, ImgstreamReader, Visualization
from STMD_cuda.models.model import Backbonev2, STFeedbackSTMD


def demo():
    # 视频调用
    objIptStream = ImgstreamReader(startImgName='D:/CodeField/Bio-Inspired detection/Dataset_linshi/v2frame/frame_0100.jpg', endImgName='D:/CodeField/Bio-Inspired detection/Dataset_linshi/v2frame/frame_0130.jpg')
    objIptStream_test = ImgstreamReader(startImgName='D:/CodeField/Bio-Inspired detection/Dataset_linshi/v2frame/frame_0100.jpg', endImgName='D:/CodeField/Bio-Inspired detection/Dataset_linshi/v2frame/frame_0130.jpg')
    # 图像帧调用
    # objIptStream = ImgstreamReader(
    #     startImgName='D:/CodeField/Bio-Inspired detection/Small-Target-Motion-Detectors-main/demodata/imgstream_67IPC/frame_0000.jpg',
    #     endImgName='D:/CodeField/Bio-Inspired detection/Small-Target-Motion-Detectors-main/demodata/imgstream_67IPC/frame_0150.jpg')
    ojbVisualize = Visualization()
    objModel = STFeedbackSTMD()
    # objModel = Backbonev2()
    objModel.init_config()

    counter = 0
    step = 1
    while objIptStream.hasFrame:
        grayImg, colorImg = objIptStream.get_next_frame()
        colorImg1 = colorImg
        if counter >= 1:
            grayImg1, colorImg1 = objIptStream_test.get_next_frame()
        # step=1时不执行
        if counter % step != 0:
            counter += 1
            print(counter)
            continue

        print(counter)

        # grayImg = cv2.resize(grayImg, (1920, 1080), interpolation=cv2.INTER_AREA)
        # colorImg = cv2.resize(colorImg, (1920, 1080), interpolation=cv2.INTER_AREA)

        result = objModel.forward(grayImg)
        frame_data = ojbVisualize.show_result(colorImg1, result)
        counter += 1


if __name__ == '__main__':
    demo()
