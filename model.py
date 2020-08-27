'''
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
*     Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
*     Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
*     Neither the names of the copyright holders nor the names of its
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
'''


import torch
import torch.nn as nn

class DistortionSensitivityModel(nn.Module):

    def __init__(self, block_size, init_steepness, init_sensitivity, 
                 distortions, optimize_gamma=False, min_steepness=0.01):
        
        """Neural network estimator for distortion sensitivity.

        Parameters
        ----------
        block_size : int
            Height and Width of input blocks. Also influences the neural 
            network architecture. (See _build().)

        init_slope : float,
            Initial value of slope parameter. 

        init_sensitivity : float,
            Initial value of the bias in the final layer. Using a good value 
            here can speed up training time and stability considerably.

        distortions : list of strings,
            Distortion types of input data. If optimize_gamma=False, this has 
            no effect. If optimize_gamma=True, the model will optimize one 
            extra parameter per distortion type that scales the network's 
            estimate of distortion sensitivity. The will be one additional 
            parameter for every distortion type in the given list.

        optimize_gamma : bool,
            If true, the model will optimize a distortion-type specific 
            parameter that scales the network's estimate of distortion 
            sensitivity.

        min_steepness : float
            Lower bound to be enforced on the steepness estimate. This will 
            stabilize the initial training phase.

        """

        super(DistortionSensitivityModel, self).__init__()
        
        self.block_size = block_size
        self.init_sensitivity = init_sensitivity
        self.init_steepness = init_steepness
        self.min_steepness = min_steepness
        self.distortions = distortions
        self.optimize_gamma = optimize_gamma

        self._build()



    def _build(self):

        """
        Sets up the model parameters. According to the paper, the neural network architecture depends on the input size (i.e., self.block_size) with:

        block_size ==   8:
        C32 C32   C64 C64   C128 C128 P C256   C256 P C512   C512 P F512 F1

        block_size ==  16:
        C32 C32   C64 C64 P C128 C128 P C256   C256 P C512   C512 P F512 F1

        block_size ==  32:
        C32 C32 P C64 C64 P C128 C128 P C256   C256 P C512   C512 P F512 F1

        block_size ==  64:
        C32 C32 P C64 C64 P C128 C128 P C256 P C256 P C512   C512 P F512 F1

        block_size == 128:
        C32 C32 P C64 C64 P C128 C128 P C256 P C256 P C512 P C512 P F512 F1

        """

        # The input-independent steepness parameter for the sigmoid function. 
        self._steepness = nn.Parameter(torch.tensor([self.init_steepness], dtype=torch.float32))    

        # Set up one scaling parameter per distortion type.
        self.gammas = dict()
        for j, distortion in enumerate(self.distortions):
            self.gammas[distortion] = nn.Parameter(torch.tensor([1.]))

        # dictorary: str -> Parameter
        self.gammas = nn.ParameterDict(self.gammas)
            

        # construct neural network
        self._features = []
        self._features.append(nn.Conv2d(1, 32, kernel_size=3, padding=1))
        self._features.append(nn.LeakyReLU(negative_slope=0.2))
        self._features.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        self._features.append(nn.LeakyReLU(negative_slope=0.2))
        if self.block_size > 8:
            # no pooling for 8x8 or 8x8 16x16 inputs
            self._features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self._features.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self._features.append(nn.LeakyReLU(negative_slope=0.2))
        self._features.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self._features.append(nn.LeakyReLU(negative_slope=0.2))
        if self.block_size > 16:
            # no pooling for 8x8 or 16x16 inputs
            self._features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self._features.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self._features.append(nn.LeakyReLU(negative_slope=0.2))
        self._features.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self._features.append(nn.LeakyReLU(negative_slope=0.2))
        self._features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self._features.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        self._features.append(nn.LeakyReLU(negative_slope=0.2))
        if self.block_size > 32:
            # no pooling for 8x8, 16x16 or 32x32 inputs
            self._features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self._features.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self._features.append(nn.LeakyReLU(negative_slope=0.2))
        self._features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self._features.append(nn.Conv2d(256, 512, kernel_size=3, padding=1))
        self._features.append(nn.LeakyReLU(negative_slope=0.2))
        if self.block_size > 64:
            # no pooling for 8x8, 16x16, 32x32 or 64x64 inputs
            self._features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self._features.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self._features.append(nn.LeakyReLU(negative_slope=0.2))
        self._features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # chain together all feature layers
        self._features = nn.Sequential(*self._features)


        self._regression = []
        self._regression.append(
            nn.Linear(in_features=512, out_features=512, bias=True))        
        self._regression.append(nn.Dropout(0.5))
        self._regression.append(
            nn.Linear(in_features=512, out_features=1, bias=True))
        self._regression[-1].bias.data.fill_(self.init_sensitivity)

        # chain together all regression layers
        self._regression = nn.Sequential(*self._regression)


    def forward(self, x):
        
        """
        Apply model to compute sensitivity.

        Parameters
        ----------
        x : (float tensor, str) tuple
            Tensor represents the input to the model and should have shape
            (batch, channel, height, width). The string specifies the 
            distortion type of the data in the batch. All data in the tensor 
            must have the same distortion type.

        Returns
        -------
        float tensor
            Estimate of distortion sensitivity per batch element.

        float tensor
            Steepness for the sigmoid regression function.
            

        """

        x, distortion = x

        sensitivity = self._regression(self._features(x).squeeze())

        if self.optimize_gamma:
            # select the distortion type-dependent scaling factor
            gamma = self.gammas[distortion]
            sensitivity = gamma * sensitivity
        
        # Early in training quality predictions are very scattered and the
        # optimal steepness is very low, which may destabilize training and 
        # corrupt computation of correlations. To avoid this, steepness is 
        # enforced to be larger than 'min_steepness' at all times. Note that
        # this usually only affects the initial training phase (perhaps 
        # first 5 iterations) after which the optimal steepness (and the 
        # estimate) should generally be larger than the enforced lower bound.
        steepness = torch.clamp(self._steepness, min=self.min_steepness)

        return sensitivity, steepness


    def get_gamma_dict(self):
        
        """
        Returns
        -------
        dict
            Mapping str -> Parameter that represents the distortion type 
            specific scaling factors gamma.
    
        """
        
        return dict(zip(self.gammas.keys(), self.gammas.values()))