#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""

# %%
from pathlib import Path
import nibabel as nib
import numpy as np
import subprocess
import argparse


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_f', help="NIfTI file")
    args = parser.parse_args()

    input_f = args.input_f

    # Squeeze dimensions
    img0 = nib.load(input_f)
    if len(img0.shape) > 4:
        V = np.squeeze(img0.get_fdata())
        img = nib.Nifti1Image(V, img0.affine)
        nib.save(img, input_f)

    # Save label
    ostr = subprocess.check_output(f"3dinfo -label {input_f}", shell=True)
    labstr = ostr.decode().rstrip().split('|')

    lab_f = Path(input_f).stem.replace('.nii', '') + '_label.txt'
    lab_f = Path(input_f).parent / lab_f
    open(lab_f, 'w').write('\n'.join(labstr))
