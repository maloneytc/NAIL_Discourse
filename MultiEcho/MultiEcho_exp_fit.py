import nibabel as nib
import pandas as pd
import numpy as np
import scipy.optimize
#import matplotlib.pyplot as plt
import sys
import pdb

from pathlib import Path
import os
import argparse

#Turn off polyfit's rank warning
import warnings
warnings.simplefilter('ignore', np.RankWarning)


# ----------------------------------------------------------------------
def polyfit(multi_echo_1_data, multi_echo_2_data, multi_echo_3_data, echo_times):
    """docstring for polyfit"""

    #TODO: Generalize to an arbitrary number of echos

    assert(len(echo_times) == 3)

    ndims = multi_echo_1_data.ndim
    if ndims == 4:
        xdim, ydim, zdim, tdim = multi_echo_1_data.shape
    elif ndims == 3:
        xdim, ydim, zdim = multi_echo_1_data.shape
        tdim = 1
        multi_echo_1_data = multi_echo_1_data.reshape((xdim, ydim, zdim, tdim))
        multi_echo_2_data = multi_echo_2_data.reshape((xdim, ydim, zdim, tdim))
        multi_echo_3_data = multi_echo_3_data.reshape((xdim, ydim, zdim, tdim))


    #def multi_echo_func(x, S, R):
    #    return S * np.exp(-R * x)

    S0_data = np.zeros_like(multi_echo_1_data, dtype=np.float)
    R2_data = np.zeros_like(multi_echo_1_data, dtype=np.float)
    T2_data = np.zeros_like(multi_echo_1_data, dtype=np.float)

    echos = [[],[],[]]
    index = []

    for t in range(0, tdim):
        message = "On volume {} of {}".format(t+1, tdim)
        sys.stdout.write(message)  # write the next character
        sys.stdout.flush()                # flush stdout buffer (actual character display)
        sys.stdout.write('\b' * len(message))
        for x in range(0, xdim):
            for y in range(0, ydim):
                for z in range(0, zdim):
                    me_vox_1 = multi_echo_1_data[x,y,z,t]
                    me_vox_2 = multi_echo_2_data[x,y,z,t]
                    me_vox_3 = multi_echo_3_data[x,y,z,t]

                    if (me_vox_1 != 0) & (me_vox_2 != 0) & (me_vox_3 != 0):
                        index.append((x,y,z,t))
                        echos[0].append(me_vox_1)
                        echos[1].append(me_vox_2)
                        echos[2].append(me_vox_3)

    xfit = echo_times
    print("Fitting...")
    R, logS0 = np.polyfit(xfit, np.log(echos), 1)

    R = -R
    S0 = np.exp(logS0)
    T2 = 1.0 / R

    for item, this in enumerate(index):
        x,y,z,t = this
        S0_data[x,y,z,t] = S0[item]
        R2_data[x,y,z,t] = R[item]

        #TODO Upper threshold is somewhat arbitrary!!! Choosen from paper by Speck although they were using a 1.5T scanner
        # Negative T2* is impossible
        if (T2[item] > 1000) or (T2[item] < 0):
            T2_data[x,y,z,t] = 0
        else:
            T2_data[x,y,z,t] = T2[item]

    S0_data = np.nan_to_num(S0_data)
    R2_data = np.nan_to_num(R2_data)
    T2_data = np.nan_to_num(T2_data)

    return S0_data, R2_data, T2_data


# ----------------------------------------------------------------------
def interleave(multi_echo_1_data, multi_echo_2_data, multi_echo_3_data):
    """docstring for interleave"""
    n_echos = 3

    xdim, ydim, zdim, tdim = multi_echo_1_data.shape

    interleaved_data = np.zeros((xdim, ydim, zdim, n_echos * tdim), dtype=multi_echo_1_data.dtype)

    interleaved_data[:,:,:,0::n_echos] = multi_echo_1_data
    interleaved_data[:,:,:,1::n_echos] = multi_echo_2_data
    interleaved_data[:,:,:,2::n_echos] = multi_echo_3_data

    return interleaved_data


# ----------------------------------------------------------------------
def separate_echos(multi_echo_data, n_echos = 3):
    """docstring for separate_echos"""

    multi_echo_1_data = multi_echo_data[:,:,:,0::n_echos]
    multi_echo_2_data = multi_echo_data[:,:,:,1::n_echos]
    multi_echo_3_data = multi_echo_data[:,:,:,2::n_echos]

    return multi_echo_1_data, multi_echo_2_data, multi_echo_3_data


# ----------------------------------------------------------------------
def average_echos(multi_echo_1_data, multi_echo_2_data, multi_echo_3_data):
    """docstring for average_echos"""
    n_echos = 3

    xdim, ydim, zdim, tdim = multi_echo_1_data.shape

    avg_data = np.zeros_like(multi_echo_1_data, dtype=multi_echo_1_data.dtype)

    avg_data = np.mean(np.array([multi_echo_1_data, multi_echo_2_data, multi_echo_3_data]), axis=4)

    #TODO can be removed after testing
    assert avg_data.shape == multi_echo_1_data.shape

    return avg_data


# ----------------------------------------------------------------------
def te_weighted_summation(echo_data_list, echo_times, t2_baseline_data):
    """docstring for te_weighted_summation"""

    xDim, yDim, zDim, tDim = echo_data_list[0].shape
    n_echos = len(echo_times)

    combined_data = np.zeros((xDim, yDim, zDim, n_echos, tDim))
    for item, echo_data in enumerate(echo_data_list):
        combined_data[:, :, :, item, :] = echo_data

    mask_indices = np.where(t2_baseline_data > 0)

    #TODO There's likely a numpy way of doing this and avoiding the loop
    #t2_baseline_masked_data = np.zeros((mask_indices[0].shape[0]))
    #for index in range(0, mask_indices[0].shape[0]):
    #    t2_baseline_masked_data[index] = t2_baseline_data[mask_indices[0][index], mask_indices[1][index], mask_indices[2][index]]
    t2_baseline_masked_data = t2_baseline_data[mask_indices]

    alpha = np.zeros((t2_baseline_masked_data.shape[0], n_echos))
    for item, echo_time in enumerate(echo_times):
        alpha[:, item] = echo_time * np.exp(-echo_time / t2_baseline_masked_data)

    alpha_sum = np.sum(alpha, axis=1)

    alpha_3D = np.zeros((xDim, yDim, zDim, n_echos))
    for echo_index in range(0, n_echos):
        alpha_3D[mask_indices[0], mask_indices[1], mask_indices[2], echo_index] = alpha[:, echo_index]/alpha_sum
    # for index in range(0, mask_indices[0].shape[0]):
    #     for echo_index in range(0, n_echos):
    #         alpha_3D[mask_indices[0][index], mask_indices[1][index], mask_indices[2][index], echo_index] = alpha[index, echo_index]/alpha_sum[index]

    alpha_3D = np.rollaxis(np.tile(alpha_3D, (tDim,1,1,1,1)),0,5)

    weighted = combined_data * alpha_3D
    data_out = np.sum(weighted, axis = 3)

    return data_out
    # optimally_combined_img = nib.Nifti1Image(data_out, img1.affine)
    # optimally_combined_img.to_filename(output_file)



# ----------------------------------------------------------------------
def run(echo_files, output_basename, echo_times=None):
    """docstring for run"""

    if len(echo_files) == 3:
        assert(len(echo_times) == 3)
        multi_echo_1_img = nib.load(str(echo_files[0]))
        multi_echo_1_data = multi_echo_1_img.get_data()

        multi_echo_2_img = nib.load(str(echo_files[1]))
        multi_echo_2_data = multi_echo_2_img.get_data()

        multi_echo_3_img = nib.load(str(echo_files[2]))
        multi_echo_3_data = multi_echo_3_img.get_data()

        affine = multi_echo_1_img.affine

    elif len(echo_files) == 1:
        multi_echo_img = nib.load(str(echo_files[0]))
        header = multi_echo_img.header
        if ('image', '.rec') in multi_echo_img.files_types:
            echo_times = np.unique(header.image_defs['echo_time'])
            echo_times.sort()
            echo_times_str = ', '.join([str(this_time) for this_time in echo_times])
            affine = multi_echo_img.affine
            print(f'PAR/REC file loaded with {len(echo_times)} echos at echo times {echo_times_str}.\n\n')
        else:
            raise Exception('The script is not currently set up to handle single echo files that are not PAR/REC.')

        multi_echo_1_data, multi_echo_2_data, multi_echo_3_data = separate_echos(multi_echo_img.get_data(), n_echos=len(echo_times))

        echo_1_img = nib.Nifti1Image(multi_echo_1_data, affine=affine)
        echo_1_img.to_filename(output_basename+'_echo1.nii.gz')

        echo_2_img = nib.Nifti1Image(multi_echo_2_data, affine=affine)
        echo_2_img.to_filename(output_basename+'_echo2.nii.gz')

        echo_3_img = nib.Nifti1Image(multi_echo_3_data, affine=affine)
        echo_3_img.to_filename(output_basename+'_echo3.nii.gz')

    else:
        raise Exception('The script is not currently set up to handle less than three echos.')

    # Fit exponential model
    # ----------------------
    S0_data, R2_data, T2_data = polyfit(multi_echo_1_data, multi_echo_2_data, multi_echo_3_data, echo_times=echo_times)

    # Write out files
    # -----------------
    #R2_img = nib.Nifti1Image(R2_data, affine=multi_echo_1_img.affine, header=multi_echo_1_img.header)
    R2_img = nib.Nifti1Image(R2_data, affine=affine)
    R2_img.to_filename(output_basename+'_R2.nii.gz')

    S0_img = nib.Nifti1Image(S0_data, affine=affine)
    S0_img.to_filename(output_basename+'_S0.nii.gz')

    T2_img = nib.Nifti1Image(T2_data, affine=affine)
    T2_img.to_filename(output_basename+'_T2.nii.gz')


    weighted_summation = te_weighted_summation([multi_echo_1_data, multi_echo_2_data, multi_echo_3_data], echo_times, T2_data)
    T2_weighted_img = nib.Nifti1Image(weighted_summation, affine=affine)
    T2_weighted_img.to_filename(output_basename+'_T2weighted.nii.gz')

    return echo_times

# ----------------------------------------------------------------------
if __name__ == "__main__":

    # Check python version
    ######################
    min_python_version = (3, 6)
    if sys.version_info < min_python_version:
        sys.exit("Python %s.%s or later is required.\n" % min_python_version)

    # Check for FSL
    ######################
    try:
        fsldir = Path(os.environ["FSLDIR"])
    except KeyError:
        sys.stderr("FSL not found, please install FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) or make sure FSLDIR environment variable is set.")
        sys.exit(1)

    # --------------------
    def validFile(inFile):
        inFile = Path(inFile)
        if inFile.exists():
            return inFile
        else:
            raise Exception("Could not locate the file: {}".format(inFile))
    # --------------------

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', metavar='input_files', type=validFile, required=True, nargs='+', help='Multi-echo input file(s).')
    parser.add_argument('-o', '--out', metavar='outputName', type=str, required=True, help="Base name for output files.")
    parser.add_argument('-t', '--te', metavar='echoTimes', type=float, required=False, default=None, help=" Echo times. E.g. 10 20 30", nargs='+')

    args = parser.parse_args()

    run(args.input, args.out, echo_times=args.te)
