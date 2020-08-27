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


from absl import app
from absl.flags import argparse_flags

import sys
import os
import argparse
import time
import json

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from model import DistortionSensitivityModel
from iqaDataFrame import IqaDataFrame 
from iqaDataset import IqaDataset, RandomBlocks, ToTensor, FlipHorizontal


def squared_difference(inputs, targets):
    """Elementwise squarred difference between input tensors."""
    return (inputs - targets) ** 2

def mse_to_paMse(mse, sensitivity, block_avg_dim):
    """
    Compute image wise paMse

    Parameters:
    -----------

    mse : tensor,
        Blockwise mse

    sensitivity : tensor,
        Blockwise estimates of distortion sensitivity. Must have same shape as 
        mse.

    block_avg_dim : int,
        Dimension across which to average the blockwise adjusted (local) 
        paMSEs.

    """
    paMSE_block = 10**(sensitivity / 10.) * mse
    return paMSE_block.mean(dim=block_avg_dim)
    

def psnr(mse, epsilon=None, peak=torch.tensor(65025.)):
    """
    Compute peak signal-to-noise ratio from mse

    Parameters:
    -----------
    
    mse : pytorch.float

    epsilon :  pytorch.float
        Stability term used as lower bound for the mse. Should be small ~0.001.

    peak : float
        Peak value for PSNR, e.g. (2**8 - 1)**2 = 255**2 = 65025.

    """
    if not epsilon is None:
        mse = torch.max(mse, epsilon)
    return 10 * (torch.log10(peak) - torch.log10(mse))



def predict(args, model=None, dataloader=None, loss_function=None, 
    device=None, return_dataFrame=True, save_dataFrame=True, 
    save_sensitivity_maps=True, modus=""):

    """
    Predict image quality. This method can be used with a trained model and is 
    also used to make validation set predictions during training. The keyword 
    arguments are used during training but do not have to be altered when 
    making predictions using a trained model.

    Parameters:
    -----------

    args : arguments Namespace as returned by argument parser,
        Arguments used for prections, see parse_args(). If any other argument 
        is specified, the other argument will be used and the value in the 
        args namespace is ignored.

    model : pytorch model,
        During training the model will be passed via this argument.

    dataloader : pytorch dataloader,
        During training the validation set's dataloader will be passed via 
        this argument.

    loss_function : pytorch loss function,
        During training the loss function will be passed via this argument.

    device : str,
        Device to use (e.g. 'cuda:0').

    return_dataFrame : bool,
        If True, a pandas.DataFrame will be returned with the model 
        predictions. Default is true, but will be set to False during training.

    save_dataFrame : bool,
        If True, the predictions will be saved under args.save_model_path as a 
        'predictions.csv'.

    save_sensitivity_maps : bool,
        If True and args.create_sensitivity_maps == True, the sensitivity maps 
        corresponding to the model predictions will be saved as .png and .npy 
        files under args.save_model_path. This argument is only useful during 
        training to prevent sensitivity maps from being saved in every 
        iteration over the validation set. (I.e. during training, this 
        argument will be set to 'False' except for the final model 
        predictions.)

    modus : str,
        Will be used as part of the path under which the sensitivity maps are 
        saved, e.g. pass 'training' to save sensitivity maps under os.path.join
        (args.save_model_path, training, *.png)

    Returns:
    --------

    if return_dataFrame == True:
    pandas.DataFrame hold ing model predictions

    if return_dataFrame == False:

    avg_loss : float
        average prediction L1 loss

    pcc : float
        pearson correlation between predictions and true quality targets

    srocc : float
        spearman correlation between predictions and true quality targets
    """


    def image_to_blocks(image, block_size):

        """
        Extract as many non-overlapping blocks as possible from a given image, 
        starting at the top left corner. The remaining pixels that do not make
        up further blocks are discarded.

        Parameters:
        -----------

        image : 4d-tensor
            Image tensor of dimensions (batch_size, channels, height, width). 
            Batch_size and channels are supposed to be 1, i.e. 
            batch_size = channels = 1.

        block_size : int
            Integer specifying height and width of blocks to be extracted.

        Returns:
        --------

        blocks : 4d-tensor
            Tensor of dimensions (blocks, 1, block_size, block_size) that 
            contains all extracted blocks.

        """
        
        b, c, h, w = image.size()
        # number of blocks that fit into image height
        numH = h//block_size
        # number of blocks that fit into image width
        numW = w//block_size
        # shrink image to multiple of block_size
        image = image[:,:,:numH * block_size, :numW * block_size]
        # after this shape will be (batch, height, width, channels)
        image = image.transpose(1,2).transpose(2,3)
        b, h, w, c = image.size()
        # reshape image to blocks
        blocks = image.reshape(
            [-1, numH, block_size, numW, block_size, 1]).transpose(
                2,3).reshape(-1, block_size, block_size, 1)
        # after this shape will be (blocks, channels, height, width)
        return blocks.transpose(2,3).transpose(1,2)


    if model is None:
        model = torch.load(args.load_model_path)
        model.optimize_gamma = args.optimize_gamma

    if device is None:
        # is a gpu available?
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.to(device)

    if loss_function is None:
        loss_function = torch.nn.L1Loss()

    if dataloader is None:
        
        # prepare dataset

        iqaDataFrame = IqaDataFrame(data_dir=args.data_dir, 
            datasets=args.dataset, distortions=args.distortion, 
            nTrain=args.nTrain, nVal=args.nVal, testIdx=args.testIdx, 
            random_seed=args.random_seed, ref_image=args.ref_image, 
            dist_image=args.dist_image)

        dataset = IqaDataset(iqaDataFrame.df_test, transform=ToTensor())
        
        dataloader = DataLoader(dataset, batch_size=1, num_workers=8)

    avg_loss = 0
    preds = []
    paPsnrs = []
    qs = []

    if return_dataFrame or save_dataFrame:
        results = pd.DataFrame(columns=["dataset", "selected_distortion", 
            "refname", "distortion", "height", "width", "fps", "block_size", 
            "random_seed", "optimize_gamma", "estimate_on_distorted", 
            "quality", "q_norm", "mse", "psnr", "papsnr", "preds", "gamma", 
            "steepness"])

        gammas = model.get_gamma_dict()

    # stability term for the blockwise mse
    epsilon = torch.tensor(0.001).to(device)

    model.eval()

    for i_batch, sample_batched in enumerate(dataloader):
        
        # send data to gpu, if available
        img_ref = sample_batched["img_ref"].to(device)
        img_dist = sample_batched["img_dist"].to(device)
        q_norm = sample_batched["q_norm"].to(device)

        if args.estimate_on_distorted:
            # switch input
            img_ref, img_dist = img_dist, img_ref

        # validation data contains full images, not blocks
        
        blocks_ref = image_to_blocks(img_ref, args.block_size)
        blocks_dist = image_to_blocks(img_dist, args.block_size)

        block_mse = squared_difference(
            blocks_ref, blocks_dist).mean(axis=(2,3))

        # apply model, shape: b, c, 1, 1
        nn_input = (blocks_ref, sample_batched["distortion"][0])
        sensitivity, steepness = model(nn_input)

        if args.create_sensitivity_maps and save_sensitivity_maps:
            create_sensitivity_maps(sensitivity.detach().cpu().numpy().reshape(
                sample_batched["height"]//args.block_size, 
                sample_batched["width"]//args.block_size), 
                title=sample_batched["refname"][0].split("/")[-1], 
                save_dir=os.path.join(args.save_model_path, "sens_maps",modus))

        # shape: (1,)
        paMSE_img = mse_to_paMse(block_mse, sensitivity, block_avg_dim=(0,1))

        # shape: (1,)
        paPSNR_img = psnr(paMSE_img, epsilon=epsilon)

        # prediction: (1,)
        prediction = torch.sigmoid(steepness*paPSNR_img)

        loss = loss_function(prediction.squeeze(), q_norm.squeeze())
        
        # track loss and predictions
        avg_loss += loss.detach().cpu().squeeze()

        paPsnrs.extend(np.atleast_1d(paPSNR_img.detach().cpu().squeeze()))

        preds.extend(np.atleast_1d(prediction.detach().cpu().squeeze()))
        
        qs.extend(np.atleast_1d(q_norm.detach().cpu().squeeze()))

        if return_dataFrame or save_dataFrame:
            
            results.loc[i_batch, "dataset"] = sample_batched["dataset"][0]

            results.loc[i_batch, "selected_distortion"] = args.distortion[0]
    
            results.loc[i_batch, "refname"] = sample_batched["refname"][0]
    
            results.loc[i_batch, "distortion"] = \
                sample_batched["distortion"][0]
            
            results.loc[i_batch, "height"] = \
                sample_batched["height"].numpy().squeeze()

            results.loc[i_batch, "width"] = \
                sample_batched["width"].numpy().squeeze()

            results.loc[i_batch, "fps"] = \
                sample_batched["fps"].numpy().squeeze()

            results.loc[i_batch, "block_size"] = args.block_size

            results.loc[i_batch, "random_seed"] = args.random_seed

            results.loc[i_batch, "optimize_gamma"] = args.optimize_gamma

            results.loc[i_batch, "estimate_on_distorted"] = \
                args.estimate_on_distorted

            results.loc[i_batch, "quality"] = \
                sample_batched["quality"].numpy().squeeze()

            results.loc[i_batch, "q_norm"] = \
                sample_batched["q_norm"].detach().cpu().squeeze().numpy()
           
            results.loc[i_batch, "mse"] = \
                sample_batched["mse"].squeeze().numpy()

            results.loc[i_batch, "psnr"] = \
                psnr(sample_batched["mse"]).squeeze().numpy()

            results.loc[i_batch, "papsnr"] = \
                paPSNR_img.detach().cpu().squeeze().numpy()

            results.loc[i_batch, "preds"] = \
                prediction.detach().cpu().squeeze().numpy()

            # If the model was not trained on the distortion type that it is 
            # tested on, the gamma will be set to np.nan. Else the optimized
            # gamma will be returned.
            _d = sample_batched["distortion"][0] 
            if _d in gammas:
                results.loc[i_batch, "gamma"] = \
                gammas[_d].detach().cpu().numpy()[0]
            else:
                results.loc[i_batch, "gamma"] = np.nan

            results.loc[i_batch, "steepness"] = \
                steepness.detach().cpu().numpy().squeeze()

    if save_dataFrame:
        
        if not os.path.exists(args.save_model_path):
            os.makedirs(args.save_model_path)

        results.to_csv(os.path.join(args.save_model_path, "predictions.csv"))


    if return_dataFrame:
        return results

    else:
        avg_loss = avg_loss/(i_batch+1)
        pcc = pearsonr(preds, qs)[0]
        srocc = spearmanr(preds, qs)[0]

        return avg_loss, pcc, srocc


def create_sensitivity_maps(sensitivity, title, save_dir):

    """
    Plot and save a sensitivity map. Will also save the sensitivity map as a 
    numpy array (.npy file).

    Parameters
    ----------

    sensitivity : 2d-array
        Sensitivity maps to be plotted.

    title : str
        Title and filename for the plot.

    save_dir : str
        Path to directory under which to save the plot
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.imshow(sensitivity)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, title + ".png"))
    plt.close()
    np.save(file=os.path.join(save_dir, title + ".npy"), arr=sensitivity)
    return


        
def parse_args(argv):
    
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', '1'):
            return True
        elif v.lower() in ('no', 'false', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    ### Generic arguments ###

    parser.add_argument("--data_dir",
        type=str, default="../datasets")

    parser.add_argument("--block_size", 
        type=int, default=32,
        help="Block size for which to estimate distortion sensitivity. "
             "Requires a single int which specifies both width and height.")

    parser.add_argument("--random_seed", 
        type=int, default=None,
        help="Random seed for tensorflow and numpy.")

    parser.add_argument("--load_model_path", 
        type=str, default=None,
        help="You may specify the path to an existing model to continue "
             "training or to use this model for predictions.")

    parser.add_argument("--dataset", 
        type=str, nargs="+",
        help="Names of datasets to be used.")

    parser.add_argument("--distortion", 
        type=str, nargs="+", required=True, 
        help="Distortion type(s) to use. Defaults to using all distortion "
             "types.")

    parser.add_argument("--nTrain", 
        type=int, default=17,
        help="Number of reference images in the training set.")

    parser.add_argument("--nVal", 
        type=int, default=6,
        help="Number of reference images in the validation set.")

    parser.add_argument("--testIdx", 
        type=int, default=None,
        help="You can select a specific reference image which (together with "
             "the corresponding distorted images) will constitute the test "
             "set. The index refers to the n-th reference image, in "
             "alphabetical order.")

    parser.add_argument("--estimate_on_distorted", 
        default=False, type=str2bool,
        help="Switch the input to the neural network: If this flag is used, "
             "the distorted image is fed into the neural network instead of "
             "the reference image.")

    parser.add_argument("--save_model_path", 
        type=str, default="./saved_models",
        help="Location under which to store the trained model and results.")

    parser.add_argument("--create_sensitivity_maps", 
        default=False, type=str2bool,
        help="Create sensitivity maps.")

    parser.add_argument("--optimize_gamma", 
        default=False, type=str2bool, 
        help="If true, the model will optimize distortion-type specific "
             "factors that scale the distortion sensitivity estimated by the "
             "neural network. A model trained with this mechanism enabled can "
             "only be evaluated on the same distortion types that were used "
             "during training. (We used this to produce the gamma-star "
             "results in the paper).")

    subparsers = parser.add_subparsers(title="command", 
        dest="command", 
        help="Choose 'train' to optimize a new model or 'predict' to use an "
             "existing model to make predictions.")


    ### train arguments ###

    train_cmd = subparsers.add_parser("train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a new model.")

    train_cmd.add_argument("--batch_size", 
        type=int, default=4,
        help="(Maximum) Number of reference images per training-batch. The "
             "actual batch-size, i.e., the number of data samples per batch, "
             "is determined by this argument in conjunction with --num_blocks "
             "as (--batch_size * --num_blocks)")

    train_cmd.add_argument("--num_blocks_per_image", 
        type=int, default=8,
        help="Number of blocks to sample from a single image.")

    train_cmd.add_argument("--n_epochs", 
        type=int, default=300,
        help="Number of training epochs.")



    ### predict arguments ###

    predict_cmd = subparsers.add_parser("predict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Load a trained model and compute prediction on new data.")

    predict_cmd.add_argument("--ref_image",
        type=str, default=None,
        help="Path to a reference image. This argument will override the "
             "generic dataset argument and is useful if you want to deploy "
             "the model on single images (or, more generally, on data that "
             "has not been formated according to the 'iqaDataFrame' class). "
             "If you use this argument, you will also need to supply a path "
             "for a distorted image via '--dist_image'.")

    predict_cmd.add_argument("--dist_image",
        type=str, default=None,
        help="Path to distorted image. See help for '--ref_image'.")

    # Parse arguments.
    args = parser.parse_args(argv[1:])

    if args.command is None:
        # should be 'train' or 'predict'
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    
    if not args.random_seed is None:
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Invoke subcommand.
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
