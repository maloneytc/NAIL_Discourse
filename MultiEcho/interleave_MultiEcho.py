import nibabel as nib
import sys
import os
import numpy as np

# ----------------------------------------------------------------------
def run(output, echos):
    """docstring for run"""

    echo1_img = nib.load(echos[0])
    echo1_data = echo1_img.get_data()
    data_shape = echo1_data.shape

    #echo2_img = nib.load(echo2)
    #echo2_data = echo2_img.get_data()

    #echo3_img = nib.load(echo3)
    #echo3_data = echo3_img.get_data()

    full_data = np.stack([nib.load(this).get_data() for this in echos], axis=-1).reshape(data_shape[0], data_shape[1], data_shape[2], -1)
    img = nib.Nifti1Image(full_data, echo1_img.affine)

    img.to_filename(output)

if __name__=="__main__":

    output = sys.argv[1]
    echos = sys.argv[2:]

    run(output, echos)


