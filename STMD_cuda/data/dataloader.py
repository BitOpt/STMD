import cv2
import os
import sys
import glob
from tkinter import filedialog
import time
import numpy as np
from scipy.spatial import cKDTree

from smalltargetmotiondetectors.util.matrixnms import MatrixNMS
from STMD_cuda.utils.debug_show import ndarray_show, ndarray_save


class VidstreamReader:
    """
    VidstreamReader - Class for reading frames from a video file.
    This class provides methods to read frames from a video file and visualize
    the progress using a waitbar.

    Properties:
        hasFrame - Indicates if there are more frames available.

    Hidden Properties:
        hVid - VideoCapture object for reading the video file.
        startFrame - Starting frame number.
        currIdx - Index of the current frame being processed.
        frameIdx - Index of the current frame in the video.
        hWaitbar - Handle to the waitbar.
        endFrame - Ending frame number.


    Methods:
        __init__ - Constructor method.
        get_next_frame - Retrieves the next frame from the video.
        __del__ - Destructor method.

    Example:
        # Create VidstreamReader object and read frames
        vidReader = VidstreamReader('video.mp4')
        while vidReader.hasFrame:
            grayFrame, colorFrame = vidReader.get_next_frame()
            # Process frames
        # Delete the VidstreamReader object
        del vidReader
    """

    def __init__(self, vidName=None, startFrame=0, endFrame=None):
        """
        Constructor method for VidstreamReader class.
        Args:
        vidName (str): Name of the video file.
        startFrame (int, optional): Starting frame number. Defaults to 1.
        endFrame (int, optional): Ending frame number. Defaults to None, which indicates the last frame of the video
        Returns:
        VidstreamReader: Instance of the VidstreamReader class.
        """
        if vidName is None:
            # Get the full path of this file
            filePath = os.path.abspath(__file__)
            # Find the index of '/smalltargetmotiondetectors/' in the file path
            indexPath = filePath.find(os.path.join(os.sep, 'smalltargetmotiondetectors'))
            # Add the path to the package containing the models
            sys.path.append(filePath[:indexPath + len('smalltargetmotiondetectors') + 1])

            vidName = filedialog.askopenfilename(
                initialdir=os.path.join(filePath[:indexPath - 6], 'demodata', 'RIST_GX010290.mp4'),
                title='Selecting a input video'
            )

        self.hVid = cv2.VideoCapture(vidName)

        self.currIdx = 0
        self.hasFrame = self.hVid.isOpened()
        self.startFrame = startFrame

        if endFrame is None:
            self.endFrame = int(self.hVid.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            if endFrame > self.hVid.get(cv2.CAP_PROP_FRAME_COUNT):
                self.endFrame = int(self.hVid.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                self.endFrame = endFrame

    def get_size(self):
        width = self.hVid.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.hVid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return width, height

    def get_next_frame(self):
        """
        Retrieves the next frame from the video.

        Returns:
            tuple: A tuple containing the grayscale and color versions of the frame.
        Raises:
            Exception: If the frame cannot be retrieved.
        """

        if self.hasFrame:
            if self.currIdx == 0:
                self.hVid.set(cv2.CAP_PROP_POS_FRAMES, self.startFrame)
                self.frameIdx = self.startFrame
            ret, colorImg = self.hVid.read()
            if not ret:
                raise Exception('Could not get the frame.')

            grayImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY).astype(float) / 255
        else:
            raise Exception('Having reached the last frame.')

        if self.frameIdx < self.endFrame - 1:
            self.hasFrame = True
        else:
            self.hasFrame = False
        self.currIdx += 1
        self.frameIdx += 1

        return grayImg, colorImg

    def __del__(self):
        """
        Destructor method.
        """
        self.hVid.release()


class ImgstreamReader:
    def __init__(self, imgstreamFormat=None, startFrame=1, endFrame=None, startImgName=None,
                 endImgName=None):
        self.hasFrame = False
        self.fileList = []
        self.currIdx = 0
        self.frameIdx = 0
        self.hWaitbar = None
        self.isShowWaitbar = False  # Flag indicating whether to show waitbar
        self.hasDeleteWaitbar = False  # Flag indicating whether waitbar has been deleted
        self.imgsteamFormat = imgstreamFormat
        self.startFrame = startFrame
        self.endFrame = endFrame  # Index of the last frame

        if startImgName and endImgName is not None:
            self.startImgName = startImgName
            self.endImgName = endImgName
            self.get_filelist_from_imgName()
        elif imgstreamFormat is not None:
            self.get_filelist_from_imgsteamformat()
        else:
            raise Exception('')

        self.get_idx()

    def get_idx(self):
        # Find start and end frame indices
        shouldFoundStart = True
        for idx in range(len(self.fileList)):
            if shouldFoundStart:
                if os.path.basename(self.fileList[idx]) == os.path.basename(self.startImgName):
                    startIdx = idx
                    shouldFoundStart = False
            else:
                if os.path.basename(self.fileList[idx]) == os.path.basename(self.endImgName):
                    endIdx = idx
                    break

        # Check if start and end frames are found
        if not 'startIdx' in locals():
            raise Exception('Cannot find the start frame.')
        if not 'endIdx' in locals():
            raise Exception('Cannot find the end frame.')

        # Set fileList to frames between startIdx and endIdx
        self.fileList = self.fileList[startIdx:endIdx]

        # Check if fileList is empty
        if len(self.fileList) == 0:
            self.hasFrame = False
        else:
            self.hasFrame = True

        # Set frameIdx to startIdx
        self.frameIdx = startIdx

        # Set endFrame to endIdx
        self.endFrame = endIdx

    def get_filelist_from_imgName(self):
        '''
            get_filelist_from_imgName

            Parameters:
            - self: Instance of the ImgstreamReader class.
        '''
        # Find the index of the dot in the start image name
        dotIndex = self.startImgName.rfind('.')
        startFolder, _ = os.path.split(self.startImgName)
        endFolder, _ = os.path.split(self.endImgName)

        if not check_same_ext_name(self.startImgName, self.endImgName):
            raise Exception('Start image has a different extension than end image.')
        if os.path.basename(startFolder) != os.path.basename(endFolder):
            raise Exception('The image stream must be in the same folder!')

        # Update fileList property with files matching the selected extension
        self.fileList = glob.glob(os.path.join(startFolder, '*' + self.startImgName[dotIndex:]))

    def get_filelist_from_imgsteamformat(self):
        '''
        Get start and end frame names from image stream format
        This method generates the start and end frame names based on the specified image stream format

        Parameters:
        - self: Instance of the ImgstreamReader class.
        - imgsteamFormat: Format string specifying the image stream format.
        - startFrame: Index of the start frame.
        - endFrame: Index of the end frame.

        Returns:
        - startImgName: Name of the start frame file.
        - endImgName: Name of the end frame file.
        '''

        # Retrieve the list of files matching the image stream format
        self.fileList = glob.glob(self.imgsteamFormat)

        # Extract the basename and extension from the specified format
        basename, ext1 = os.path.splitext(self.imgsteamFormat)
        basename = basename[:-1]  # Remove the trailing *

        # Check if any files match the specified format
        if not self.fileList:
            raise Exception('No files matching the format could be found.')
        else:
            # Determine the end frame index
            if not self.endFrame:
                self.endFrame = len(self.fileList)
            else:
                self.endFrame = min(self.endFrame, len(self.fileList))

            # Extract the names of the first and last files in the list
            nameFirst = os.path.splitext(self.fileList[0])[0]
            nameEnd = os.path.splitext(self.fileList[-1])[0]

            # Determine if the file names have the same length
            if len(nameFirst) == len(nameEnd):
                # Extract the numeric part from the end file name
                num1 = self.fileList[-1].replace(basename, '').replace(ext1, '')
                numDigits1 = len(num1)

                # Generate the start and end frame names with zero-padding
                self.startImgName = "{}{:0{}}{}".format(basename, self.startFrame, numDigits1, ext1)
                self.endImgName = "{}{:0{}}{}".format(basename, self.endFrame, numDigits1, ext1)
            else:
                # Generate the start and end frame names without zero-padding
                self.startImgName = "{}{}{}".format(basename, self.startFrame, numDigits1, ext1)
                self.endImgName = "{}{}{}".format(basename, self.endFrame, numDigits1, ext1)

    def get_next_frame(self):
        '''
        get_next_frame - Retrieves the next frame from the image stream.
          This method retrieves the next frame from the image stream and returns
          both grayscale and color versions of the frame. It updates the internal
          state to point to the next frame in the stream.

          Parameters:
              - self: Instance of the ImgstreamReader class.

          Returns:
              - garyImg: Grayscale version of the retrieved frame.
              - colorImg: Color version (RGB) of the retrieved frame.
        '''

        # Get information about the current frame
        fileInfo = self.fileList[self.currIdx]

        # Try to read the image file
        try:
            colorImg = cv2.imread(fileInfo)
            self.hasFrame = True
        except:
            # If an error occurs while reading the image, set hasFrame to false
            self.hasFrame = False
            raise Exception('Could not read the image.')

        # Convert the color image to grayscale
        garyImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY).astype(float) / 255

        # Update internal state to point to the next frame
        if self.currIdx < len(self.fileList) - 1:
            self.hasFrame = True
            self.currIdx = self.currIdx + 1
            self.frameIdx = self.frameIdx + 1
        else:
            # If the end of the image stream is reached, set hasFrame to false
            self.hasFrame = False

        return np.double(garyImg), colorImg


class Visualization:
    def __init__(self, showThreshold=0.1):
        self.showThreshold = showThreshold
        self.paraNMS = {'maxRegionSize': 15,
                        'method': 'sort'}
        self.shouldNMS = True
        self.hNMS = MatrixNMS(self.paraNMS['maxRegionSize'], self.paraNMS['method'])
        self.timeTic = time.time()
        self.colorImg = None
        self.detectresult_path = 'D:/CodeField/Bio-Inspired detection/research/Detect_Result/detection_rate'
        self.counter = 0

    def show_result(self, colorImg=None, result={'response': None, 'direction': None}):
        self.colorImg = colorImg
        elapsedTime = time.time() - self.timeTic
        # print(elapsedTime)
        motionDirection = result['direction']

        modelOpt = np.array(np.squeeze(result['response'].to('cpu')))
        self.counter += 1

        if modelOpt is not None and np.any(modelOpt):
            maxOutput = np.max(modelOpt)

            if maxOutput > 0:
                if self.shouldNMS:
                    nmsOutput = self.hNMS.nms(modelOpt)
                idX, idY = np.where(nmsOutput > self.showThreshold * maxOutput)

                if motionDirection is not None:
                    # 调整到运动矢量的前一个位置(x2,y2)->(x1,y1)
                    targetLocation = np.column_stack((idX, idY))
                    motionDirection = np.array(motionDirection)[:, 2:]
                    motionDirection[:, 0] -= motionDirection[:, 3]
                    motionDirection[:, 1] -= motionDirection[:, 2]

                    tree = cKDTree(targetLocation)
                    distances, indices = tree.query(motionDirection[:, :2])
                    for i, element in enumerate(indices):  # i对应的是motion的索引，element对应的是target的索引
                        targetLocation[element][0] += motionDirection[i][3]
                        targetLocation[element][1] += motionDirection[i][2]

                    targetLocation = targetLocation[indices]

                    for x, y in zip(targetLocation[:, 1], targetLocation[:, 0]):
                        self.colorImg = cv2.rectangle(self.colorImg, (x - 15, y - 15), (x + 15, y + 15),
                                                      color=(0, 0, 255), thickness=2)
                else:
                    for x, y in zip(idY, idX):
                        self.colorImg = cv2.rectangle(self.colorImg, (x - 5, y - 5), (x + 7, y + 7),
                                                      color=(0, 0, 255), thickness=2)

        resize_colorImg = cv2.resize(self.colorImg, (1280, 720), interpolation=cv2.INTER_LINEAR)
        #vid1# if self.counter == 50 or self.counter == 130 or self.counter == 200 or self.counter == 283 or self.counter == 459:
        ndarray_save(os.path.join(self.detectresult_path, str(self.counter).zfill(4)+'.jpg'), self.colorImg)
        # ndarray_save(os.path.join(self.detectresult_path, str(self.counter).zfill(4)+'resize720p.jpg'), resize_colorImg)
        # resize_colorImg = self.colorImg
        # cv2.imshow("result", resize_colorImg)
        # cv2.waitKey(1)

        self.timeTic = time.time()

        # if modelOpt is not None and np.any(modelOpt):
        #     if maxOutput > 0:
        #         return [idX, idY]
        #     else:
        #         return None
        # else:
        #     return None


def check_same_ext_name(startImgName, endImgName):
    _, ext1 = os.path.splitext(startImgName)
    _, ext2 = os.path.splitext(endImgName)
    # Check if the extensions of the start and end images are the same
    if ext1 != ext2:
        return False
    else:
        return True
