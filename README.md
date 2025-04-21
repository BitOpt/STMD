# Bio-Inspired Small Target Detecting Visual Neural Network With Motion Direction Decoding Compensation in Large Scene
## Overview 
**MDDC-STMD** is a comprehensive package designed for detecting small moving target in large and complex backgrounds.
We provide a range of functionalities and tools to facilitate the detection and tracking of small moving targets in both image sequences and videos. A real-time solution will be provided later.
The software package is written in Python, and the convolution part of the network uses PyTorch based Cuda to accelerate the calculation to ensure low processing time per frame. Future support for C++ is planned. 
## Features
1. Provide advanced algorithms and models for small moving target detection under complex and large background.
2. Provide API for easy integration into different projects.
3. Contains tools for data preprocessing, visualization, and evaluation.
4. Support parameter configuration for specific application scenarios.
5. Include DEMO scripts to show usage and functionality.
## Package Structure
* data: Contains API functions and classes for dataset preprocessing and visualization.
* model: Contains models and neural networks used in small target motion detection.
* utils: Provides the output visualization of each module in the network.
* test: Includes demonstration scripts showcasing the usage of the package.
## How to Use:
### Installation:
Clone or download the repository and follow the setup instructions for your preferred programming language. Configure the Python virtual environment according to [requirement.txt](https://github.com/BitOpt/STMD/blob/main/STMD_cuda/requirement.txt)
### Examples:
Running [test.py](https://github.com/BitOpt/STMD/blob/main/STMD_cuda/test.py) or [newmodel_core.py](https://github.com/BitOpt/STMD/blob/main/STMD_cuda/models/newmodel_core.py)  via Python
## Support and Feedback
If you encounter any issues or have any suggestions while using the STMD package, feel free to reach out to me.
You can raise issues or submit feedback on the GitHub repository or email [3220225094@bit.edu.cn]. I will respond promptly and strive to address your concerns.
