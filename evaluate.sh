# !/bin/bash

# The copyright in this software is being made available under this Software
# Copyright License. This software may be subject to other third party and
# contributor rights, including patent rights, and no such rights are
# granted under this license.
# Copyright (c) 1995 - 2020 Fraunhofer-Gesellschaft zur Förderung der
# angewandten Forschung e.V. (Fraunhofer)
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted for purpose of testing the functionalities of
# this software provided that the following conditions are met:
# *     Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# *     Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# *     Neither the names of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, INCLUDING
# WITHOUT LIMITATION THE PATENTS OF THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS, ARE GRANTED BY THIS SOFTWARE LICENSE. THE
# COPYRIGHT HOLDERS AND CONTRIBUTORS PROVIDE NO WARRANTY OF PATENT
# NON-INFRINGEMENT WITH RESPECT TO THIS SOFTWARE.

###############################################################################

# Example 1: 
# Evaluate a pre-trained model on the jpeg subset of the csiq dataset ##
# You have to make sure that the settings for block_size and optimize_gamma 
# match the settings used for training the model.

data_dir="./datasets"
dataset="csiq"
distortion="jpeg"
nTrain="18"
nVal="6"
block_size="32"
est_on_distorted="False"
random_seed="1"

output_dir="./predictions"
path_to_model="./models/csiq/jpeg/on_reference/opt_gamma=False/32/1/dsm"

python main.py --data_dir "$data_dir" \
               --random_seed "$random_seed" \
               --dataset "$dataset" \
               --distortion "$distortion" \
               --nTrain "$nTrain" \
               --nVal "$nVal" \
               --block_size "$block_size" \
               --save_model_path "$output_dir" \
               --create_sensitivity_maps "True" \
               --estimate_on_distorted "False" \
               --optimize_gamma "False" \
               --load_model_path "$path_to_model" \
               predict

# Predictions will be saved as a csv file under '$output_dir'.
      

###############################################################################


# Example 2: 
# Evaluate a pre-trained model on novel data (not liveiqa, tid2013 or csiq) ##
# You have to make sure that the settings for block_size and optimize_gamma 
# match the settings used for training the model.

# This example assumes that the reference and distorted images are available 
# under the following paths:
path_to_ref_img="./datasets/csiq_1600_512x512_1_ref.png"
path_to_dist_img="./datasets/csiq_1600_512x512_1_jpeg_0_20.png"

distortion="some_distortion"
nTrain="18"
nVal="6"
block_size="32"
est_on_distorted="False"
random_seed="1"

output_dir="./predictions_novel"
path_to_model="./models/csiq/jpeg/on_reference/opt_gamma=False/32/1/dsm"




python main.py --distortion "$distortion" \
               --block_size "$block_size" \
               --save_model_path "$output_dir" \
               --create_sensitivity_maps "True" \
               --estimate_on_distorted "False" \
               --optimize_gamma "False" \
               --load_model_path "$path_to_model" \
               predict \
               --ref_image "$path_to_ref_img" \
               --dist_image "$path_to_dist_img"

# Predictions will be saved as a csv file under '$output_dir'.