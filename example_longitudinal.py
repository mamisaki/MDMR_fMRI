#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MDMR example for single session data.
@author: mmisaki@librad.laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import os
import sys
import re
import pickle
import datetime
import time
import subprocess
from scipy.stats import zscore

import numpy as np
import pandas as pd
import nibabel as nib
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

import MDMR


# %% 1. Setup =================================================================

# Working directory
work_dir = Path('example').resolve()

# data list file
dala_list_f = work_dir / 'DataList.csv'

# Mask file directory
mask_dir = work_dir / 'masks'

# Brain mask; brain mask
brain_mask = mask_dir / 'MNI152_T1_2mm_brain_mask.nii.gz'
aseg = mask_dir / 'aseg.nii.gz'

# downsampling voxel size (mm)
dxyz = 4


# %% --- Read data list. Create working directories ---
# Read data list
dala_list = pd.read_csv(dala_list_f)

# Set preprocessed files
src_fnames = [work_dir / f for f in dala_list.image_file]

# Preare directories
img_dir = work_dir / 'resample_img'
if not img_dir.is_dir():
    img_dir.mkdir()

connMtx_dir = work_dir / 'conn_mtx'
if not connMtx_dir.is_dir():
    connMtx_dir.mkdir()

out_dir = work_dir / 'stats'
if not out_dir.is_dir():
    out_dir.mkdir()


# %% 2. Mask and downsampling =================================================
# %%  --- Align mask to input images ---
overwrite = False

# Gray matter mask; Gray matter mask from FreeSurfer aseg
gray_mask = mask_dir / 'MNI152_T1_2mm_gm_mask.nii.gz'
if not gray_mask.is_file() or overwrite:
    excludeIds = [0, 1, 2, 4, 5, 6, 7, 14, 15, 24,
                  40, 41, 43, 44, 45, 46, 72, 31, 63, 77,
                  251, 252, 253, 254, 255]
    excIdStr = ','.join([f'{v}' for v in excludeIds])
    cmd = f"3dcalc -overwrite -a {aseg} -expr '1-amongst(a,{excIdStr})'"
    cmd += f" -prefix {gray_mask}; "
    cmd += f"3drefit -view tlrc -space MNI {gray_mask}"
    subprocess.call(cmd, shell=True)

# Resample aseg gm mask in function image resolution
gray_mask_res = mask_dir / 'MNI152_T1_2mm_gm_mask_res.nii.gz'
if not gray_mask_res.is_file() or overwrite:
    cmd = "3dfractionize -overwrite -clip 0.0"
    cmd += f" -template {brain_mask} -input {gray_mask}"
    cmd += f" -prefix {gray_mask_res}; "
    cmd += f"3drefit -view tlrc -space MNI {gray_mask_res}"
    subprocess.call(cmd, shell=True)


# %% --- Downsampling function images ---
""" Fill zero at out of the gray_mask_res and downsampling in dxyz-mm size """
overwrite = False

res_fnames = []
for si, srcf in enumerate(src_fnames):
    fbase = srcf.name
    fbase = fbase.replace('.nii', f'.res{str(dxyz)}mm.nii')
    outf = img_dir / fbase
    res_fnames.append(outf)

    if not outf.is_file() or overwrite:
        print(f"({si+1}/{len(src_fnames)}) Mask+resample image {srcf} ... ",
              end='')
        sys.stdout.flush()

        # Mask
        tmpout = srcf.name
        tmpout = outf.parent / tmpout.replace('.nii', '_gmmask.nii')
        cmd = f"3dcalc -overwrite -a {srcf} -b {gray_mask_res}"
        cmd += f" -expr 'a*step(b)' -prefix {tmpout}; "

        # resample
        cmd += f"3dresample -overwrite -rmode Cu -dxyz {dxyz} {dxyz} {dxyz}"
        cmd += f" -inset {tmpout} -prefix {outf}; rm {tmpout}"

        subprocess.call(cmd, shell=True)

        print("done")
        sys.stdout.flush()


# %% --- Down-sampling gray matter mask ---
overwrite = False

gray_mask_res_ds = mask_dir / f'MNI152_T1_2mm_gm_mask_res_{str(dxyz)}mm.nii.gz'
if not gray_mask_res_ds.is_file() or overwrite:
    cmd = "3dfractionize -overwrite -clip 0.5 -template {res_fnames[0]}"
    cmd += " -input {gray_mask_res} -prefix {gray_mask_res_ds}"
    subprocess.call(cmd, shell=True)

# intersect of signal stdev mask
sigSD_mask_ds = mask_dir / f'sigSDMask_{str(dxyz)}mm.nii.gz'
if not sigSD_mask_ds.is_file() or overwrite:
    # Make SD images
    SDimg_dir = img_dir / 'SDimage'
    if not SDimg_dir.is_dir():
        SDimg_dir.mkdir()

    sdfiles = []
    CmdLines = []
    JobNames = []
    for si, srcf in enumerate(res_fnames):
        fbase = srcf.name
        sdf = SDimg_dir / fbase.replace('errts', 'std')
        sdfiles.append(sdf)
        if not sdf.is_file() or overwrite:
            print(f"({si+1}/{len(res_fnames)}) SD image for {srcf} ... ",
                  end='')
            sys.stdout.flush()

            cmd = f"3dTstat -overwrite -stdev -mask {gray_mask_res_ds}"
            cmd += f" -prefix {sdf} {srcf}"
            subprocess.call(cmd, shell=True)

            print("done")
            sys.stdout.flush()

    # Overlap of SD images
    cmd = '3dmask_tool -overwrite'
    cmd += ' -input ' + ' '.join([str(f) for f in sdfiles])
    cmd += f' -prefix {sigSD_mask_ds} -frac 1.0'
    subprocess.call(cmd, shell=True)

# Create MDMR mask
mask_MDMR = mask_dir / f'MDMRmask_{str(dxyz)}mm.nii.gz'
if not mask_MDMR.is_file() or overwrite:
    cmd = f"3dcalc -overwrite -a {gray_mask_res_ds} -b {sigSD_mask_ds}"
    cmd += f" -expr 'step(a)*step(b)' -prefix {mask_MDMR}"
    subprocess.call(cmd, shell=True)


# %% 3. Make connectivity matrices from downsampled images ====================
overwrite = False

st = time.time()
print("-" * 80)
print(f"Making connectivity matrix with {str(dxyz)}mm-resampled data",
      end='')
print(f" ({time.ctime(st)})")
sys.stdout.flush()

mask_flat = []
maskV = nib.load(str(mask_MDMR)).get_fdata()
ConnMtx = [''] * len(res_fnames)
for si, srcf in enumerate(res_fnames):
    dataid = srcf.stem.replace('.nii', '')
    print(f"({si+1}/{len(res_fnames)}) {dataid}: ")
    sys.stdout.flush()

    dfile = connMtx_dir / f'ConnMtx.{dataid}.npy'
    if dfile.is_file() and not overwrite:
        # Load connectivity matrix from file
        if dfile.stat().st_mtime > os.stat(srcf).st_mtime:
            # Get npy file access
            try:
                ConnMtx[si] = np.load(str(dfile), mmap_mode='r')
                print(f"  Get memory-mapped data for {dfile}")
                continue
            except Exception:
                pass

    # Make connectivity matrix
    print(f"  Making connectivity matrix from {srcf} ...", end='')
    sys.stdout.flush()
    MDMR.connectivity_matrix(str(srcf), str(dfile), maskV)
    ConnMtx[si] = np.load(str(dfile), mmap_mode='r')
    print(" done")
    sys.stdout.flush()

et = time.time()
dstr = time.ctime(et)
tt = str(datetime.timedelta(seconds=et-st)).split('.')[0]
print(f"Finished ({dstr}, took {tt})\n")
sys.stdout.flush()


# %% 4. Make design matrix: X =================================================
R = robjects.r
pandas2ri.activate()

# --- Set model equation ---
model = 'Session*Diagnosis+Sex+Age+Motion+SubjID-1'

# --- Extract xdata from dala_list ---
varnames0 = np.setdiff1d(np.unique(re.split(r'[\+|\*|:|-]', model)), '1')
xdata = dala_list[varnames0]

# normalize Age, Motion variable
if 'Age' in xdata.columns:
    xdata.Age = zscore(xdata.Age)
if 'Motion' in xdata.columns:
    xdata.Motion = zscore(xdata.Motion)

# --- Make design matrix on R ---
R.assign('xdata', pandas2ri.py2rpy(xdata))
R('xdata$Session <- relevel(factor(xdata$Session), ref="1")')
R('xdata$Diagnosis <- relevel(factor(xdata$Diagnosis), ref="HC")')

R('options(width=200)')
mmrcmd = f"model.matrix(~{model}, xdata"
mmrcmd += ', contrasts = list(Sess = "contr.sum"))'
X = R(mmrcmd)

# Get variable names
varnames = []
corcmd = f"capture.output({mmrcmd})"
ostr = R(corcmd)
for ll in ostr:
    if ll and ll[0] == ' ' and re.match(r'\s+[\[\d\]]', ll) is None:
        varnames.extend(ll.strip().split())

for ni, vn in enumerate(varnames):
    if ':' in vn:
        varnames[ni] = vn.replace(':', 'x')

# --- Set nuisance variables ---
nuisance = np.zeros(len(varnames), dtype=np.bool)
nuisance[np.argwhere(['Sex' in v for v in varnames]).ravel()] = True
nuisance[np.array(varnames) == 'Age'] = True
nuisance[np.array(varnames) == 'Motion'] = True
nuisance[np.array(['SubjID' in v for v in varnames])] = True

# --- Set exchangability block ---
""" permutation will be done within a subject (s1, s2 exchange), and then
    permute SubjID as a whole-block
"""
exchBlk = {}
# Each subject
blk = np.zeros(X.shape[0])
subjVars = np.array(varnames)[np.array(['SubjID' in v for v in varnames])]
for si, v in enumerate(subjVars):
    blk[np.nonzero(X[:, varnames.index(v)])[0]] = si+1
blk[blk == 0] = si+2
exchBlk['within_block'] = blk
exchBlk['whole_block'] = blk

# Set in reg dict
reg = {}
reg['X'] = X
reg['varnames'] = varnames
reg['nuisance'] = nuisance
reg['exchBlk'] = exchBlk


# %% 5. Run MDMR ==============================================================
overwrite = False

permnum = 10000  # number of permutation

respkl = out_dir / 'MDMR_Fstat_example.pkl'
if not respkl.is_file() or overwrite:

    st = datetime.datetime.now()
    print("="*80)
    print(f"=== MDMR analysis (start at {str(st).split('.')[0]}) ===")
    sys.stdout.flush()

    # Extract variables
    X = reg['X']
    varnames = reg['varnames']
    nuisance = reg['nuisance']
    exchBlk = reg['exchBlk']

    # Run MDMR
    F, pF, Fperm, maskrm = \
        MDMR.run_MDMR(ConnMtx, X, varnames, nuisance, permnum=permnum,
                      exchBlk=exchBlk,  metric='euclidean', chunk_size=None)

    # Save result values
    Fperm_npy = {}  # permutation results in npy file
    for lab in Fperm.keys():
        npyf = respkl.name.replace('.pkl', f'_Fperm_{lab}.npy')
        np.save(str(respkl.parent / npyf), Fperm[lab])
        Fperm_npy[lab] = npyf

    # mask volume
    mask_img = nib.load(str(mask_MDMR))
    mask_aff = mask_img.affine
    maskV = mask_img.get_data()
    if len(maskrm) and maskrm.ndim == 1:
        # Update mask by excluding voxels with NaN
        mask_flat = maskV.flatten()
        mask_ix = np.nonzero(mask_flat)[0]
        rmix = mask_ix[maskrm]
        mask_flat[rmix] = 0
        maskV = np.reshape(mask_flat, mask_img.shape)

    MDMRres = {'F': F, 'Fperm_npy': Fperm_npy, 'p': pF, 'maskV': maskV,
               'aff': mask_aff}

    with open(respkl, 'wb') as fd:
        pickle.dump(MDMRres, fd)

    et = datetime.datetime.now()
    dstr = str(et).split('.')[0]
    tt = str(et-st).split('.')[0]
    print("Finished (%s, took %s)\n" % (dstr, tt))
    sys.stdout.flush()


# %% --- Save results in NIfTI file ---
overwrite = True

stat_f = out_dir / 'MDMR_Fstat_example.nii.gz'
if not stat_f.is_file() or overwrite:
    # Load MDMR results
    with open(respkl, 'rb') as fd:
        MDMRres = pickle.load(fd)

    F, pF, maskV, mask_aff = \
        MDMRres['F'], MDMRres['p'], MDMRres['maskV'], MDMRres['aff']

    # Save in NIfTI volume
    MDMR.save_map_volume(str(stat_f), F, pF, maskV, mask_aff,
                         pthrs=[0.005, 0.001])


# %% 6. Cluster size permutation test =========================================
overwrite = True

# Load MDMR results
with open(respkl, 'rb') as fd:
    MDMRres = pickle.load(fd)

F, Fperm_npy, maskV = MDMRres['F'], MDMRres['Fperm_npy'], MDMRres['maskV']

for k in Fperm_npy.keys():
    Fperm_npy[k] = respkl.parent / Fperm_npy[k]

# Check if the process complete
out_prefix = str(out_dir / 'MDMR_example')
if not overwrite:
    comp = True
    labs = sorted(F.keys())
    for ni in range(len(labs)):
        lab1 = re.sub(r'\d\d_', '', labs[ni])
        ofile = out_prefix + 'ClusterThrPerm.'
        ofile += '%s.txt' % lab1
        ofile_pdist = ofile.replace('ClusterThrPerm.',
                                    'Cluster_pdist.')
        ofile_pdist = ofile_pdist.replace('.txt', '.pkl')
        if not os.path.isfile(ofile) or \
                not os.path.isfile(ofile_pdist):
            comp = False
            break
else:
    comp = False

if not comp:
    MDMR.cluster_permutation(F, Fperm_npy, maskV, OutPrefix=out_prefix, NN=1,
                             pthrs=[0.005, 0.001], athrs=[0.05, 0.01])
