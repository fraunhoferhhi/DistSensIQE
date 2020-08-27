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


import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import curve_fit
from skimage.metrics import mean_squared_error
import sklearn.feature_extraction.image

import argparse
import os
from shutil import copyfile
import glob

class IqaDataFrame():

    def __init__(self, data_dir, datasets, nTrain, nVal, testIdx=None, 
        distortions=[], random_seed=None, ref_image=None, dist_image=None):

        """
        This class acts like an adapter for the LIVE IQA database, TID2013 
        database and the CSIQ IQA database. It will bring these databases into 
        a consistent structure and format that is represented via a pandas 
        DataFrame. This class can also be used to construct a DataFrame for a 
        single pair of reference and distorted paths.

        Parameters:
        -----------

        data_dir : str,
            Path to directory where the datasets are stored.

        datasets : list,
            List of names of datasets to be used to construct a DataFrame. The 
            data is searched for under the 'data_dir' path.

        nTrain : int,
            Number of reference images to be used in the training set.
            The test set will contain all reference images that are not used in
            the training and validation set.

        nVal : int, 
            Number of reference images to be used in the validation set. The 
            test set will contain all reference images that are not used in
            the training and validation set.

        testIdx : int,
            Select only a single reference image for the test set. The 
            reference image are indexed alphabetically, i.e. testIdx = 0 
            corresponds to the reference image whose name comes first in the 
            alphabet.

        distortions : list,
            List of strings that specify the distortions to be used. If the 
            list contains the string 'all', all distortion types will be used.

        random_seed : int,
            numpy random seed.

        ref_image : str,
            This argument needs to be used in conjunction with the 'dist_image'
            argument and are supposed to specify the paths to a reference and 
            a distorted image respectively. When these arguments are speficied 
            a dummy dataset containing only this pair of images is constructed.
            This may be helpful if you want to make quality predictions on 
            images that are not part of LIVE, TID2013 or CSIQ.

        dist_image : str,
            See 'ref_image'.


        ### dataFrame representation ###

        Training, validation and test sets are stored as dataFrames as 
        instance properties with names df_train, df_val and df_test. Each of 
        these dataFrames has the following columns:

        "dataset" (e.g. liveiqa)
        "refname" (e.g. sailing1)
        "distortion" (e.g. jpeg)
        "height" (height in pixels)
        "width" (width in pixels
        "fps" (frames per second, always '1' for images)
        "path_ref" (path to reference image, relative to 'data_dir')
        "path_dist" (path to distorted image, relative to 'data_dir')
        "mse" (mean squared error of full image in RGB colorspace)
        "quality" (quality annotation as given by database)
        "q_norm" (normalized target quality, see below)


        ### Quality normalizations ###

        The target quality scores of a database are normalized as 

        q_norm = (quality - min(quality)) / (max(quality) - min(quality))

        Min and Max are taken across an entire dataset, i.e. all reference 
        images and distortion types at once (e.g. across all images in liveiq).

        In case of DMOS, q_norm is further inverted as 
        q_norm <= 1 - q_norm

        After normalization target qualities (q_norm) are all within the range 
        [0, 1] are are positively correlated with PSNR (=negatively correlated 
        with MSE --> higher difference implies lower quality score.)

        """

        if not random_seed is None:
            np.random.seed(random_seed)

        if ref_image is None:

            # use known dataset (liveiqa, tid2013 or csiq)

            self.data_dir = data_dir

            df = pd.DataFrame()

            for ds in datasets:

                path = os.path.join(self.data_dir, ds, "{}.csv".format(ds))

                if os.path.exists(path):

                    # dataset has already been formated, we can directly load 
                    # the corresponding csv file

                    df = df.append(pd.read_csv(path, index_col=0),
                                   ignore_index=True)

                else:

                    # dataset has not been used before, we need to format it 
                    # first. You will need to manually download the dataset, 
                    # see the corresponding description in _format_liveiqa(), 
                    # _format_tid2013() or _format_csiq().

                    path = self._format_dataset(ds)

                    df = df.append(pd.read_csv(path), ignore_index=True)

            df = self._set_dtypes(df)

            # cells will hold absolute path
            df.path_ref = df.path_ref.apply(
                lambda x: os.path.join(data_dir, x))

            df.path_dist = df.path_dist.apply(
                lambda x: os.path.join(data_dir, x))

            if not "all" in distortions:
                # keep only selected distortion types
                df = df.loc[df.distortion.isin(distortions)]

            refs = sorted(df.refname.unique())

            if len(refs) < nTrain + nVal:
                raise ValueError("Not enough references ({}) to create "
                    "training and validation sets of sizes nTrain={} and "
                    "nVal={}.".format(len(refs), nTrain, nVal))

            if not testIdx is None:

                # use testIdx-th reference for the test set

                if len(refs) != nTrain + nVal + 1:
                    raise Warning("Not all references are used. (There are {} "
                        "references. You selected nTrain={}, nVal={} and "
                        "testIdx={} (so nTest=1))".format(
                            len(refs), nTrain, nVal, testIdx))

                refs = np.roll(refs, testIdx)
                refs_test = refs[0]
                refs_train_and_val = refs[1:]
                np.random.shuffle(refs_train_and_val)
                refs_train = refs_train_and_val[:nTrain]
                refs_val = refs_train_and_val[nTrain:nTrain + nVal]


            else:

                # select random reference(s) for the test set
                np.random.shuffle(refs)
                refs_train = refs[:nTrain]
                refs_val = refs[nTrain:nTrain + nVal]
                refs_test = refs[nTrain + nVal:]

                if len(refs_test) < 1:
                    raise Warning("There are {} references. With nTrain={} "
                        "and nVal={}, the test set is empty.".format(
                            len(refs), nTrain, nVal))

            self.df_train = df.loc[df.refname.isin(refs_train)]
            self.df_val = df.loc[df.refname.isin(refs_val)]
            self.df_test = df.loc[df.refname.isin(refs_test)]

            set_refs_train = set(self.df_train.refname.unique())
            set_refs_val = set(self.df_val.refname.unique())
            set_refs_test = set(self.df_test.refname.unique())

            assert set_refs_train.intersection(set_refs_val) == set(), \
                "Some references are in training and validation set with " \
                "train={} vs val={}".format(set_refs_train, set_refs_val)

            assert set_refs_train.intersection(set_refs_test) == set(), \
                "Some references are in training and test set with train={} " \
                "vs val={}".format(set_refs_train, set_refs_test)

            assert set_refs_val.intersection(set_refs_test) == set(), \
                "Some references are in validation and test set with " \
                "train={} vs val={}".format(set_refs_val, set_refs_test)


        else: # ref_image is not None

            # empty training set
            self.df_train = pd.DataFrame(columns=
                ["dataset", "refname", "distortion", "height", "width", "fps",
                 "path_ref", "path_dist", "mse", "quality", "q_norm"])

            # empty validation set
            self.df_val = pd.DataFrame(columns=
                ["dataset", "refname", "distortion", "height", "width", "fps",
                 "path_ref", "path_dist", "mse", "quality", "q_norm"])

            # construct dummy dataset with only 1 reference image and 1 
            # distorted image. The data will be stored in self.df_test
            ref_img = np.array(Image.open(ref_image).convert("L"))
            dist_img = np.array(Image.open(dist_image).convert("L"))

            mse = np.mean((ref_img - dist_img)**2)
            h, w = ref_img.shape

            df_test = pd.DataFrame()
            df_test.loc[0, "dataset"] = "custom"
            df_test.loc[0, "refname"] = ref_image
            df_test.loc[0, "distortion"] = distortions
            df_test.loc[0, "height"] = h
            df_test.loc[0, "width"] = w
            df_test.loc[0, "fps"] = 1
            df_test.loc[0, "path_ref"] = ref_image
            df_test.loc[0, "path_dist"] = dist_image
            df_test.loc[0, "mse"] = mse
            df_test.loc[0, "quality"] = np.nan
            df_test.loc[0, "q_norm"] = np.nan

            self.df_test = self._set_dtypes(df_test)

            

    def _set_dtypes(self, df):
        return df.astype({"dataset": str,
                          "refname": str,
                          "distortion": str,
                          "height": int,
                          "width": int,
                          "fps": int,
                          "path_ref": str,
                          "path_dist": str,
                          "mse": float,
                          "quality": float,
                          "q_norm": float})



    def fit_sigmoid(self):

        """
        Fit parameters c and d of the sigmoid function f(x)=1/exp(-c * (x-d)) 
        on the psnr and quality targets (q_norm) values of the data in the 
        training set (self.df_train). 

        Returns:
        --------

        steepness : float,
            optimal c

        shift : float,
            optimal d
        """

        def sigmoid(x, a, b, c, d):
            return a + (b-a) / np.exp(-c * (x-d))

        psnrs = 10. * np.log10(255**2 / self.df_train.mse.values)
        q = self.df_train.q_norm.values

        [steepness, shift], cov = curve_fit(lambda x, c, d: 
            sigmoid(x, 0, 1, c, d), psnrs, q, p0=[0.1, 29.], method="dogbox")

        return steepness, shift


    def _format_dataset(self, ds):

        """
        Format dataset.

        The formated dataset will have the following structure:

        data_dir
           |
           |--<ds> (e.g. liveiqa)
               |
               |--reference
                    |
                    |--ref_image 1
                    |--ref_image 2
                    |--...
                    |--ref_image N
               |
               |--<distortion type1> (e.g. jpeg)
                    |--dist_image 1
                    |--dist_image 2
                    |--...
                    |--dist_image M
               |
               |--<distortion type1> (e.g. jp2k)
                    |--dist_image 1
                    |--dist_image 2
                    |--...
                    |--dist_image L
               |
               ...

        """

        if ds == "liveiqa":
            return self._format_liveiqa()

        elif ds == "tid2013":
            return self._format_tid2013()

        elif ds == "csiq":
            return self._format_csiq()

        raise ValueError("Unknown dataset: {}".format(ds))


    def _format_csiq(self):

        """
        Format the CSIQ dataset.

        You need to manually download the dataset from 
        http://vision.eng.shizuoka.ac.jp/mod/page/view.php?id=23

        Specifically, you need to download the files 'src_imgs.zip', 
        'dst_imgs.zip' and 'csiq.DMOS.xlsx'.

        Assuming '/data' is the directory where you store datasets, you need 
        to create a directory 'csiq', place the files in this this directory 
        and extract the zip files.

        The directory should have the following structure:

        /data
            |
            --csiq
                |
                --dst_imgs.zip
                --dst_imgs
                --src_imgs.zip
                --src_imgs
                --csiq.DMOS.xlsx

        After formatting (, that is, after running this method), the directory 
        should have the following structure:

        /data
            |
            --csiq
                |
                --dst_imgs.zip
                --dst_imgs
                --src_imgs.zip
                --src_imgs
                --csiq.DMOS.xlsx
                --awgn
                --contrast
                --fnoise
                --gblur
                --jp2k
                --jpeg
                --reference
                --csiq.csv

        Reference images will be located in 'reference'. Distorted images will be located in awgn, contrast, fnoise, gblur, jp2k and jpeg. 

        File names are altered to the following pattern: 
        (reference images:)
        csiq_<reference name>_<resolution>_<fps>_ref.bmp
        (distorted images:)
        csiq_<reference name>_<resolution>_<fps>_<distortion>_<quality>.bmp

        csiq.csv stores all information about the images and will be used 
        to construct a iqaDataFrame object.

        """

        def get_dist_type(x):
            if x == "jpeg": return "jpeg", "jpeg"
            elif x == "jpeg 2000": return "jpeg2000", "jp2k"
            elif x == "blur": return "blur", "gblur"
            elif x == "fnoise": return "fnoise", "fnoise"
            elif x == "noise": return "awgn", "awgn"
            elif x == "contrast": return "contrast", "contrast"
            else:
                raise ValueError("Unknown distortion type: {}".format(x))

        source = os.path.join(self.data_dir, "csiq")

        scores = pd.read_excel(os.path.join(source, "csiq.DMOS.xlsx"), 
            sheet_name="all_by_image", header=3, usecols=[3, 5, 6, 8])

        for path in glob.glob(os.path.join(source, "src_imgs", "*")):
            os.rename(path, path.lower())

        for path in glob.glob(os.path.join(source, "dst_imgs", "*", "*")):
            os.rename(path, path.lower())

        if not os.path.exists(os.path.join(source, "reference")):
            os.makedirs(os.path.join(source, "reference"))

        db = pd.DataFrame(columns=["dataset", "refname", "distortion",
                                   "height", "width", "fps",
                                   "path_ref", "path_dist", "mse", "quality"])

        for i, row in scores.iterrows():

            print("\rFormating csiq: {:4d}/{:4d}".format(
                i+1, scores.shape[0]), end="")

            distType_file, distType_new = get_dist_type(row.dst_type)

            # original name of reference image
            img_name_ref = "{}.png".format(row.image)
            # new path for reference image
            path_img_ref_new = glob.glob(
                os.path.join(source, "reference", "*_{}_*".format(row.image)))

            if len(path_img_ref_new) == 0:
                 # reference image has not been copied to new directory yet
                 img_ref = np.array(
                    Image.open(os.path.join(source, "src_imgs", img_name_ref)
                        ).convert("L"))
                 
                 h, w = img_ref.shape[0:2]
                 
                 img_name_ref_new = "csiq_{}_{}x{}_1_ref.png".format(
                    row.image, h, w)

                 os.rename(os.path.join(source, "src_imgs", img_name_ref),
                           os.path.join(source, "reference", img_name_ref_new))
            
            elif len(path_img_ref_new) == 1:
                # reference image has already been copied
                img_ref = np.array(Image.open(path_img_ref_new[0]
                    ).convert("L"))
                
                h, w = img_ref.shape[0:2]
                
                img_name_ref_new = "csiq_{}_{}x{}_1_ref.png".format(
                    row.image, h, w)
            else:
                raise ValueError("Multiple reference images found for glob "\
                    "expression '*_{}_*'".format(row.image))

            img_name_dist = "{}.{}.{}.png".format(
                row.image, distType_file, row.dst_lev)

            img_dist = np.array(Image.open(os.path.join(
                source, "dst_imgs", distType_file, img_name_dist)
                ).convert("L"))

            img_name_dist_new = "csiq_{}_{}x{}_1_{}_{:03.2f}.png".format(
                row.image, h, w, distType_new, row.dmos)

            # create directory for this distortion type
            if not os.path.exists(os.path.join(source, distType_new)):
                os.makedirs(os.path.join(source, distType_new))

            # copy distorted image
            os.rename(
                os.path.join(source, "dst_imgs", distType_file, img_name_dist),
                os.path.join(source, distType_new, img_name_dist_new))

            # compute mse
            mse = mean_squared_error(img_ref, img_dist)

            rowIdx = db.shape[0]
            db.loc[rowIdx, "dataset"] = "csiq"
            db.loc[rowIdx, "refname"] = row.image
            db.loc[rowIdx, "distortion"] = distType_new
            db.loc[rowIdx, "height"] = h
            db.loc[rowIdx, "width"] = w
            db.loc[rowIdx, "fps"] = 1
            db.loc[rowIdx, "path_ref"] = \
                os.path.join("csiq", "reference", img_name_ref_new)
            
            db.loc[rowIdx, "path_dist"] = \
                os.path.join("csiq", distType_new, img_name_dist_new)
            
            db.loc[rowIdx, "mse"] = mse
            db.loc[rowIdx, "quality"] = row.dmos

        # normalize quality scores
        q_range = db.quality.max() - db.quality.min()
        db.loc[:, "q_norm"] = 1 - (db.quality - db.quality.min()) / q_range

        path_csv = os.path.join(source, "csiq.csv")

        db.to_csv(path_csv)

        print("Formating csiq: done")

        return path_csv


    def _format_tid2013(self):

        """
        Format the TID 2013 dataset.

        You will need to manually download the dataset from 
        http://www.ponomarenko.info/tid2013.htm

        Specifically, you need to download the file 'tid2013.rar'.

        Assuming '/data' is the directory where you store datasets, move the file to this directory and extract it.

        The directory structure should look as follows afterwards:

        /data
            |
            --tid2013.rar
            --tid2013
                |
                --distorted_images
                --metrics_values
                --papers
                --reference_images
                --mos.txt
                --mos_std.txt
                --mos_with_names.txt
                --readme

        After formatting (, that is, running this method,) the directory 
        should look as follows:

        /data
            |
            --tid2013.rar
            --tid2013
                |
                --distorted_images
                --metrics_values
                --papers
                --reference_images
                --mos.txt
                --mos_std.txt
                --mos_with_names.txt
                --readme
                --awgn
                --awgn2
                --ca
                --cc
                --ccs
                --cn
                --gblur
                --hfn
                --icqd
                --id
                --in
                --jp2k
                --jp2kt
                --lbdi
                --lcni
                --mgn
                --mn
                --ms
                --nepn
                --qn
                --reference
                --scn
                --ssr
                --tid2013.csv

        All references will be located in 'reference'. Distorted images will 
        be located in awgn, awgn2, ..., ssr.

        File names are altered to the following pattern: 
        (reference images:)
        tid2013_<reference name>_<resolution>_<fps>_ref.bmp
        (distorted images:)
        tid2013_<reference name>_<resolution>_<fps>_<distortion>_<quality>.bmp

        tid2013.csv stores all information about the images and will be used 
        to construct a iqaDataFrame object.

        """

        def index2distortion(idx):

            if idx == 1: return "awgn"
            if idx == 2: return "awgn2"
            if idx == 3: return "scn"
            if idx == 4: return "mn"
            if idx == 5: return "hfn"
            if idx == 6: return "in"
            if idx == 7: return "qn"
            if idx == 8: return "gblur"
            if idx == 9: return "id"
            if idx == 10: return "jpeg"
            if idx == 11: return "jp2k"
            if idx == 12: return "jpegt"
            if idx == 13: return "jp2kt"
            if idx == 14: return "nepn"
            if idx == 15: return "lbdi"
            if idx == 16: return "ms"
            if idx == 17: return "cc"
            if idx == 18: return "ccs"
            if idx == 19: return "mgn"
            if idx == 20: return "cn"
            if idx == 21: return "lcni"
            if idx == 22: return "icqd"
            if idx == 23: return "ca"
            if idx == 24: return "ssr"
            
            raise ValueError("Unknown distortion index: {}".format(idx))

        source = os.path.join(self.data_dir, "tid2013")

        for path in glob.glob(os.path.join(source, "reference_images", "*")):
            os.rename(path, path.lower())

        for path in glob.glob(os.path.join(source, "distorted_images", "*")):
            os.rename(path, path.lower())

        if not os.path.exists(os.path.join(source, "reference")):
            os.makedirs(os.path.join(source, "reference"))

        db = pd.DataFrame(columns=["dataset", "refname", "distortion",
                                   "height", "width", "fps",
                                   "path_ref", "path_dist", "mse", "quality"])

        with open(os.path.join(source, "mos_with_names.txt"), "r") as mos_file:

            for i, line in enumerate(mos_file):
                
                print("\rFormating tid2013: {:4d}".format(i + 1), end="")

                mos, filename = line.rstrip("\n").lower().split(" ")

                mos = float(mos)

                path_img_dist = glob.glob(
                    os.path.join(source, "distorted_images", filename))[0]

                name, idx, level = filename.split(".")[0].split("_")
                dist = index2distortion(int(idx))

                path_img_ref = glob.glob(
                    os.path.join(source, "reference_images", name+".bmp"))[0]

                img_ref = np.array(Image.open(path_img_ref).convert("L"))
                img_dist = np.array(Image.open(path_img_dist).convert("L"))

                mse = mean_squared_error(img_ref, img_dist)
                h, w = img_ref.shape

                new_name_ref = "tid2013_{}_{}x{}_1_ref.bmp".format(name, h, w)
                new_name_dist = "tid2013_{}_{}x{}_1_{}_{:03.2f}.bmp".format(
                    name, h, w, dist, mos)

                if not os.path.exists(
                    os.path.join(source, "reference", new_name_ref)):

                    copyfile(path_img_ref, 
                        os.path.join(source, "reference", new_name_ref))

                if not os.path.exists(os.path.join(source, dist)):
                    os.makedirs(os.path.join(source, dist))

                os.rename(path_img_dist, os.path.join(source, dist, 
                    new_name_dist))

                rowIdx = db.shape[0]
                db.loc[rowIdx, "dataset"] = "tid2013"
                db.loc[rowIdx, "refname"] = name
                db.loc[rowIdx, "distortion"] = dist
                db.loc[rowIdx, "height"] = h
                db.loc[rowIdx, "width"] = w
                db.loc[rowIdx, "fps"] = 1
                db.loc[rowIdx, "path_ref"] = \
                    os.path.join("tid2013", "reference", new_name_ref)
                
                db.loc[rowIdx, "path_dist"] = \
                    os.path.join("tid2013", dist, new_name_dist)
                
                db.loc[rowIdx, "mse"] = mse
                db.loc[rowIdx, "quality"] = mos

            print()

            q_range = db.quality.max() - db.quality.min()

            db.loc[:, "q_norm"] = (db.quality - db.quality.min()) / q_range

            path_csv = os.path.join(source, "tid2013.csv")

            db.to_csv(path_csv)

            print("Formating tid2013: done")

            return path_csv


    def _format_liveiqa(self):

        """
        Format the LIVE IQA dataset.

        You will need to manually download the dataset from 
        https://live.ece.utexas.edu/research/Quality/subjective.htm

        Specifically, you need to download the 'Subjective database Release 2' 
        (the file is called 'databaserelease2.zip') and realigned subjective quality data (which comes in a separate file called 
        'dmos_realigned.mat').

        Assuming '/data' is your default location where you store datasets, 
        please create a directory called 'liveiqa', place the downloaded files 
        in it and extract the zip file. The directory structure should 
        look as follows afterwards:

        /data
           |
           --liveiqa
                |
                --databaserelease2
                --databaserelease2.zip
                --dmos_realigned.mat

        After formatting (, that is, after successfully running this method), the directory will look as follows:

        /data
           |
           --liveiqa
                |
                --awgn
                --databaserelease2
                --fastfading
                --gblur
                --jp2k
                --jpeg
                --reference
                --databaserelease2.zip
                --dmos_realigned.mat
                --liveiqa.csv

        All references will be located in 'reference'. Distorted images will 
        be located in awgn, fastfading, gblur, jp2k and jpeg.

        File names are altered to the following pattern: 
        (reference images:)
        liveiqa_<reference name>_<resolution>_<fps>_ref.bmp
        (distorted images:)
        liveiqa_<reference name>_<resolution>_<fps>_<distortion>_<quality>.bmp

        liveiqa.csv stores all information about the images and will be used 
        to construct a iqaDataFrame object.
        """

        def index2distortion(idx):
            if idx < 227:
                return "jp2k", 0, "jp2k"
            elif idx < 227 + 233:
                return "jpeg", 227, "jpeg"
            elif idx < 227 + 233 + 174:
                return "wn", 227 + 233, "awgn"
            elif idx < 227 + 233 + 174 + 174:
                return "gblur", 227 + 233 + 174, "gblur"
            else:
                return "fastfading", 227 + 233 + 174 + 174, "fastfading"

        source = os.path.join(self.data_dir, "liveiqa", "databaserelease2")
        destination = os.path.join(self.data_dir)

        dmoses = loadmat(
            os.path.join(self.data_dir, "liveiqa", "dmos_realigned.mat"))
        dmoses = dmoses["dmos_new"].squeeze()
        names = np.hstack(np.hstack(loadmat(
            os.path.join(source, "refnames_all.mat"))["refnames_all"]))

        db = pd.DataFrame(columns=["dataset", "refname", "distortion",
                                   "height", "width", "fps",
                                   "path_ref", "path_dist", "mse", "quality"])

        for i, (name, dmos) in enumerate(zip(names, dmoses)):

            print("\rFormating liveiqa: {:4d}/{:4d}".format(
                i + 1, len(dmoses)), end="")

            if dmos == 0:
                # reference image
                continue

            # get distortion type
            dist, offset, newDist = index2distortion(i)

            # load image
            ref_img = Image.open(
                os.path.join(source, "refimgs", name)).convert("L")
            dist_img = Image.open(
                os.path.join(source, dist, "img{}.bmp".format(i + 1 - offset))
                    ).convert("L")

            # get resolution
            res_x = ref_img.size[0]
            res_y = ref_img.size[1]

            # compute mse
            mse = mean_squared_error(np.array(ref_img), np.array(dist_img))

            # create directory for reference images
            if not os.path.exists(
                os.path.join(destination, "liveiqa", "reference")):
                
                os.makedirs(os.path.join(destination, "liveiqa", "reference"))

            # create directory for this distortion type
            if not os.path.exists(
                os.path.join(destination, "liveiqa", newDist)):
                
                os.makedirs(os.path.join(destination, "liveiqa", newDist))

            # construct new names
            new_img_name = name.split(".")[0].replace("_", "").lower()

            new_name_ref = "liveiqa_{}_{}x{}_1_ref.bmp".format(
                new_img_name, res_x, res_y)

            new_name_dst = "liveiqa_{}_{}x{}_1_{}_{:03.2f}.bmp".format(
                new_img_name, res_x, res_y, newDist, dmos)

            new_path_ref = os.path.join("liveiqa", "reference", new_name_ref)
            new_path_dist = os.path.join("liveiqa", newDist, new_name_dst)

            # copy reference image
            if not os.path.exists(os.path.join(destination, new_path_ref)):
                copyfile(os.path.join(source, "refimgs", name), 
                    os.path.join(destination, new_path_ref))

            # copy distorted image
            if not os.path.exists(os.path.join(destination, new_path_dist)):
                os.rename(os.path.join(source, dist, "img{}.bmp".format(
                    i + 1 - offset)), os.path.join(destination, new_path_dist))

            rowIdx = db.shape[0]
            db.loc[rowIdx, "dataset"] = "liveiqa"
            db.loc[rowIdx, "refname"] = new_img_name
            db.loc[rowIdx, "distortion"] = newDist
            db.loc[rowIdx, "height"] = res_y
            db.loc[rowIdx, "width"] = res_x
            db.loc[rowIdx, "fps"] = 1
            db.loc[rowIdx, "path_ref"] = new_path_ref
            db.loc[rowIdx, "path_dist"] = new_path_dist
            db.loc[rowIdx, "mse"] = mse
            db.loc[rowIdx, "quality"] = dmos

        q_range = db.quality.max() - db.quality.min()

        db.loc[:, "q_norm"] = 1 - (db.quality - db.quality.min()) / q_range

        path_csv = os.path.join(destination, "liveiqa", "liveiqa.csv")

        db.to_csv(path_csv)

        print("Formating liveiqa: done")

        return path_csv


if __name__ == "__main__":

    # useful for debugging

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, nargs="+", required=True)
    parser.add_argument("--distortion", type=str, nargs="+", required=True)
    parser.add_argument("--dataset", "-ds", type=str, default="liveiqa")
    parser.add_argument("--nTrain", type=int, default=17)
    parser.add_argument("--nVal", type=int, default=6)
    parser.add_argument("--testIdx", type=int)

    args = parser.parse_args()

    iqedata = IqaDataFrame(data_dir=args.data_dir, datasets=args.dataset, 
        distortions=args.distortion, nTrain=args.nTrain, nVal=args.nVal)