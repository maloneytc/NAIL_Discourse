import nibabel as nib
import sys
import os
import numpy as np
from pathlib import Path
import pdb

# ----------------------------------------------------------------------
def run(input, n_echos):
    """docstring for run"""

    echo_img = nib.load(input)
    echo_data = echo_img.get_fdata()
    data_shape = echo_data.shape

    for this_echo in range(0, n_echos):
        this_data = echo_data[:,:,:,this_echo::n_echos]
        this_output = input.rstrip('.gz').rstrip('.nii') + f'_e{this_echo+1}.nii.gz'
        img = nib.Nifti1Image(this_data, echo_img.affine)
        img.to_filename(this_output)

# ----------------------------------------------------------------------
if __name__=="__main__":

    input = sys.argv[1]
    n_echos = int(sys.argv[2])

    run(input, n_echos)
