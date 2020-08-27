This repository provides an implementation for our paper

## Estimation of distortion sensitivity for visual quality prediction using a convolutional neural network
(https://www.sciencedirect.com/science/article/pii/S1051200418308868)

```
article{bosse2019estimation,
  title={Estimation of distortion sensitivity for visual quality prediction using a convolutional neural network},
  author={Bosse, Sebastian and Becker, S{\"o}ren and M{\"u}ller, Klaus-Robert and Samek, Wojciech and Wiegand, Thomas},
  journal={Digital Signal Processing},
  volume={91},
  pages={54--65},
  year={2019},
  publisher={Elsevier}
}
```

### Code structure
The code is written in python and uses pytorch. Required packages and the versions we used are listed in 'requirements.txt'. The implementation consists of 4 files:

#### iqaDataFrame.py
iqaDataFrame.py acts like an adapter for the various image quality datasets that brings them into a consistent format. Formating is currently implemented for the liveIQA, TID2013, and CSIQ datasets. Please have a look at the documentation in _format_liveiqa(), _format_tid2013() and _format_csiq() on how to use this class.

#### iqaDataset.py
iqaDataset.py transforms an iqaDataFrame into a dataset that can be used with a pytorch dataloader. 

#### model.py
model.py implements the archticture of the convolutional neural network.

#### main.py
You can use this file to evaluate a pre-trained model. Pre-trained models for various distortion types can be found in the models directory. Please have a look at evaluate.sh to see how to use this code to evaluate a pre-trained model. 

### Training your own model
If you would like to obtain the code for training a model, please contact Sören Becker (soeren.becker at hhi.fraunhofer.de).

### License

The copyright in this software is being made available under this Software
Copyright License. This software may be subject to other third party and
contributor rights, including patent rights, and no such rights are
granted under this license.
Copyright (c) 1995 - 2020 Fraunhofer-Gesellschaft zur Förderung der
angewandten Forschung e.V. (Fraunhofer)
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted for purpose of testing the functionalities of
this software provided that the following conditions are met:

\*     Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

\*     Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

\*     Neither the names of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, INCLUDING
WITHOUT LIMITATION THE PATENTS OF THE COPYRIGHT HOLDERS AND
CONTRIBUTORS, ARE GRANTED BY THIS SOFTWARE LICENSE. THE
COPYRIGHT HOLDERS AND CONTRIBUTORS PROVIDE NO WARRANTY OF PATENT
NON-INFRINGEMENT WITH RESPECT TO THIS SOFTWARE.
