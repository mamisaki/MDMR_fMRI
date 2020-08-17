# -*- coding: utf-8 -*-
"""
Multivariate Distance Matrix Regression (MDMR)
@author: mmisaki@laureateinstitute.org
"""


# %% future ###################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# %% import ###################################################################
from pathlib import Path
import sys
import time
import datetime
import re
import subprocess
import multiprocessing
import pickle
import gc
from six import string_types

import numpy as np
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
from scipy import ndimage
import nibabel as nib


# %% ##########################################################################
def _count_clustsize(statmap, NN=1):
    """Count cluster size

    Parameters
    ----------
    statmap: 3D array
        (thresholded) statistical map to find clusters
    NN: integer, optional
        cluster definition code
        1; faces touch
        2; faces or edges touch
        3; faces or edges or corners touch
        default NN=1

    Returns
    -------
    cluster_sizes: array
        cluster size distribution
    """

    if NN == 1:
        NNstruct = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                             [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    elif NN == 2:
        NNstruct = np.array([[[0, 1, 0], [0, 1, 0], [0, 1, 0]],
                             [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                             [[0, 1, 0], [0, 1, 0], [0, 1, 0]]])
    elif NN == 3:
        NNstruct = np.ones((3, 3, 3))

    cluster_map, n_clusters = ndimage.label(statmap, structure=NNstruct)
    cluster_sizes = np.bincount(cluster_map.ravel())[1:]
    if len(cluster_sizes) == 0:
        cluster_sizes = [1]

    del statmap

    return cluster_sizes


# %% ##########################################################################
def connectivity_matrix(fname, outfname, mask=None, SVCmask=None,
                        cast_float32=True):
    """ Calculate Fisher's z-transformed corelation matrix (upper triangle
        part) and save the result in file.
        If there are censored volumes, those are excluded from correlation
        calculation.

    Parameters
    ----------
    fname: string
        filename of time-course image (4D BRIK or NIfTI)
    outfname: string
        Save result in nunpy binary data (.npy file)
    mask: string or numerical array (optional)
        whole-brain (all gray matter) mask for MDMR
        string: mask filename (3D BRIK or NIfTI)
        or
        numerical array: mask data
        If no mask is provided, voxels with all 0 time-course voxels are
        masked, assuming that the common mask across subjects has already been
        applied.
     SVCmask: string or numerical array (optional)
        MDMR mask for small volume correction (SVC)
        string: mask filename (3D BRIK or NIfTI)
        or
        numerical array: mask data
        If no mask is provided, voxels with all 0 time-course voxels are
        masked, assuming that the common mask across subjects has already been
        applied.

    cast_float32: bool (optional, True in default)
        Cast data to float32 for saving memory

    Return
    ------
    No return varibale but the result is saved in outfname file.
    The saved data is numpy array of half triangle of Fisher's z-transformed
    corelation matrix between voxels (in the mask).
    Use scipy.spatial.distance.squareform to reconstruct a full matrix.
    """

    srcV = nib.load(fname).get_data()
    Nt = srcV.shape[3]
    src_flat_all = np.reshape(srcV, [-1, Nt])

    # Mask
    if mask is None:
        mask_flat =\
            np.nonzero(np.logical_not(np.all(src_flat_all == 0, axis=1)))[0]
    elif isinstance(mask, string_types):
        maskV = nib.load(mask).get_data()
        mask_flat = maskV.flatten()
    elif type(mask) == np.ndarray:
        if not np.all(mask.shape == srcV.shape[:3]):
            errmsg = "Mask dimension %s" % str(mask.shape)
            errmsg += " mismatch with time-course image %s" \
                % str(srcV.shape[:3])
            assert np.all(mask.shape == srcV.shape[:3]), errmsg

        mask_flat = mask.flatten()

    # SVC Mask
    if SVCmask is not None:
        if isinstance(SVCmask, string_types):
            SVCmaskV = nib.load(SVCmask).get_data()
            SVCmask_flat = SVCmaskV.flatten()
        elif type(SVCmask) == np.ndarray:
            if not np.all(SVCmask.shape == srcV.shape[:3]):
                errmsg = "SVC mask dimension %s" % str(SVCmask.shape)
                errmsg += " mismatch with time-course image %s" \
                    % str(srcV.shape[:3])
                assert np.all(SVCmask.shape == srcV.shape[:3]), errmsg

            SVCmask_flat = SVCmask.flatten()

    # Masked data
    src_flat = src_flat_all[mask_flat > 0, :]
    if SVCmask is not None:
        SVC_src_flat = src_flat_all[SVCmask_flat > 0, :]

    del src_flat_all

    # Remove censored volumes
    rmvi = []
    rmvi = np.nonzero(np.all(src_flat == 0, axis=0))[0]
    if len(rmvi):
        src_flat = np.delete(src_flat, rmvi, axis=1)
        if SVCmask is not None:
            SVC_src_flat = np.delete(SVC_src_flat, rmvi, axis=1)

    # Connectivity
    if SVCmask is None:
        # All voxel x voxel
        # 1 - (correlation pdst) = 1-(1-r) = r
        connmtx = 1-pdist(src_flat, 'correlation')
    else:
        # voxels in SVC mask x all voxels
        allz = zscore(src_flat, axis=1)
        svcz = zscore(SVC_src_flat, axis=1)
        connmtx = np.dot(svcz, allz.T)/svcz.shape[1]
        connmtx[np.abs(connmtx) >= 1.0] = np.nan

    zconnmtx = np.arctanh(connmtx)
    del connmtx
    zconnmtx[np.isinf(zconnmtx)] = np.nan
    if cast_float32:
        zconnmtx = zconnmtx.astype(np.float32)

    np.save(outfname, zconnmtx)


# %% ##########################################################################
def PPI_connectivity_matrix(fname, outfname, blkReg, mask=None, SVCmask=None,
                            cast_float32=True):
    """ Calculate Fisher's z-transformed corelation matrix (upper triangle
        part) with PPI regressor (blkReg * seedts) and save the result in file.

        If there are censored volumes, those are excluded from correlation
        calculation.

    Parameters
    ----------
    fname: string
        filename of time-course image (4D BRIK or NIfTI)
    outfname: string
        Save result in nunpy binary data (.npy file)
    blkReg: array
        block timecourse regressor
    mask: string or numerical array (optional)
        whole-brain (all gray matter) mask for MDMR
        string: mask filename (3D BRIK or NIfTI)
        or
        numerical array: mask data
        If no mask is provided, voxels with all 0 time-course voxels are
        masked, assuming that the common mask across subjects has already been
        applied.
     SVCmask: string or numerical array (optional)
        MDMR mask for small volume correction (SVC)
        string: mask filename (3D BRIK or NIfTI)
        or
        numerical array: mask data
        If no mask is provided, voxels with all 0 time-course voxels are
        masked, assuming that the common mask across subjects has already been
        applied.

    cast_float32: bool (optional, True in default)
        Cast data to float32 for saving memory

    Return
    ------
    No return varibale but the result is saved in outfname file.
    The saved data is numpy array of half triangle of Fisher's z-transformed
    corelation matrix between voxels (in the mask).
    Use scipy.spatial.distance.squareform to reconstruct a full matrix.
    """

    srcVimg = nib.load(str(fname))
    assert srcVimg.shape[-1] == len(blkReg), \
        f"Missmatch data length ({srcVimg.shape[-1]}: {fname.name})" + \
        f" from blkReg ({len(blkReg)})"

    srcV = srcVimg.get_data()
    Nt = srcV.shape[3]
    src_flat_all = np.reshape(srcV, [-1, Nt])

    # Mask
    if mask is None:
        mask_flat =\
            np.nonzero(np.logical_not(np.all(src_flat_all == 0, axis=1)))[0]
    elif isinstance(mask, string_types):
        maskV = nib.load(mask).get_data()
        mask_flat = maskV.flatten()
    elif type(mask) == np.ndarray:
        if not np.all(mask.shape == srcV.shape[:3]):
            errmsg = "Mask dimension %s" % str(mask.shape)
            errmsg += " mismatch with time-course image %s" \
                % str(srcV.shape[:3])
            assert np.all(mask.shape == srcV.shape[:3]), errmsg

        mask_flat = mask.flatten()

    # SVC Mask
    if SVCmask is not None:
        if isinstance(SVCmask, string_types):
            SVCmaskV = nib.load(SVCmask).get_data()
            SVCmask_flat = SVCmaskV.flatten()
        elif type(SVCmask) == np.ndarray:
            if not np.all(SVCmask.shape == srcV.shape[:3]):
                errmsg = "SVC mask dimension %s" % str(SVCmask.shape)
                errmsg += " mismatch with time-course image %s" \
                    % str(srcV.shape[:3])
                assert np.all(SVCmask.shape == srcV.shape[:3]), errmsg

            SVCmask_flat = SVCmask.flatten()

    # Masked data
    src_flat = src_flat_all[mask_flat > 0, :]
    if SVCmask is not None:
        SVC_src_flat = src_flat_all[SVCmask_flat > 0, :]

    del src_flat_all

    # Remove censored volumes
    rmvi = []
    rmvi = np.nonzero(np.all(src_flat == 0, axis=0))[0]
    if len(rmvi):
        src_flat = np.delete(src_flat, rmvi, axis=1)
        if SVCmask is not None:
            SVC_src_flat = np.delete(SVC_src_flat, rmvi, axis=1)

        blkReg = np.delete(blkReg, rmvi)

    # Connectivity
    if SVCmask is None:
        # All voxel x voxel
        PPI_src_flat = zscore(src_flat * blkReg, axis=1)
        z_src_flat = zscore(src_flat, axis=1)
        connmtx = np.dot(PPI_src_flat, z_src_flat.T)/len(blkReg)
        triu_idx = np.triu_indices(connmtx.shape[0], 1)
        connmtx = connmtx[triu_idx]
    else:
        # voxels in SVC mask x all voxels
        allz = zscore(src_flat, axis=1)
        PPI_svcz = zscore(np.dot(SVC_src_flat, blkReg), axis=1)
        connmtx = np.dot(PPI_svcz, allz.T)/len(blkReg)
        connmtx[np.abs(connmtx) >= 1.0] = np.nan

    zconnmtx = np.arctanh(connmtx)
    del connmtx
    zconnmtx[np.isinf(zconnmtx)] = np.nan
    if cast_float32:
        zconnmtx = zconnmtx.astype(np.float32)

    np.save(outfname, zconnmtx)


# %% ##########################################################################
def _slice_vectorized_dmtx(vdata, row=None, col=None, mdim=None,
                           rmdiag=True):
    """ Get slice of distance matrix from vectorized data

    Parameters
    ----------
    vdata: numpy array
        vectorized distance matrix
    row: array
        reading row indices
    col: array
        reading column indices
    mdim: int
        size of distance matrix
    rmdiag: bool
        whether to remove diagonal emements

    Returns
    -------
    out_mtx: array
        len(row) x len(col) (or len(row) x (len(col)-1) if rmdiag==True),
        matrix extracted from distance matrix.
    """

    if mdim is None:
        mdim = int(np.ceil(np.sqrt(vdata.shape[0] * 2)))
        if mdim * (mdim - 1) / 2 != int(vdata.shape[0]):
            raise ValueError('Incompatible vector size.')

    if row is None:
        row = np.arange(mdim)
    else:
        row = np.array(row)

    if col is None:
        col = np.arange(mdim)
    else:
        col = np.array(col)

    if rmdiag:
        outshape = [len(row), len(col)-1]
    else:
        outshape = [len(row), len(col)]

    out_mtx = np.zeros(outshape, dtype=vdata.dtype)
    # ri < ci area
    for iir, ri in enumerate(row):
        stp = int(np.sum(range(mdim-1, mdim-1-ri, -1)))
        widx = np.nonzero(col > ri)[0]
        if len(widx) == 0:
            continue

        rdcols = col[widx]-(ri+1) + stp
        if rmdiag:
            out_mtx[iir, widx-1] = vdata[rdcols]
        else:
            out_mtx[iir, widx] = vdata[rdcols]

    # ri > ci area
    for iic, ci in enumerate(col):
        stp = int(np.sum(range(mdim-1, mdim-1-ci, -1)))
        widx = np.nonzero(row > ci)[0]
        if len(widx) == 0:
            continue

        rdrows = row[widx]-(ci+1) + stp
        out_mtx[widx, iic] = vdata[rdrows]

    return out_mtx


# %% ##########################################################################
def _check_nan_connectivity(ConnMts, verb=True):
    """Check nan connectivity
    """

    if verb:
        st = time.time()
        print("+++ Check NaN in connectivity maps (%s)" % time.ctime(st))
        sys.stdout.flush()

    dmask = None  # nan mask
    for si, cmtx in enumerate(ConnMts):
        if verb:
            print("\r    %d/%d ..." % (si+1, len(ConnMts)), end='')
            sys.stdout.flush()

        if dmask is None:
            dmask = np.isnan(cmtx)
        else:
            dmask = np.logical_or(dmask, np.isnan(cmtx))

    if dmask.ndim < 2:
        # reconstract to matrix
        dmaskMtx = squareform(dmask).astype(np.bool)
        np.fill_diagonal(dmaskMtx, True)

        # find rows (voxels) with all nan (True in dmaskMtx)
        delvi = np.nonzero(np.all(dmaskMtx, axis=1))[0]
        del dmaskMtx
    else:
        # find nan indices (True in dmaskMtx)
        delvi = np.nonzero(dmask)
        delvi = np.array([[i, j] for i, j in zip(delvi[0], delvi[1])])

    if verb:
        print(' done')

    return delvi


# %% ##########################################################################
def run_MDMR(ConnMts, X, regnames=[], nuisance=[], contrast={}, permnum=10000,
             exchBlk=[], metric='euclidean', chunk_size=None, verb=True):
    """
    Multivariate Distance Matrix Regression (MDMR)

    Parameters
    ----------
    ConnMts : list of ndarray
        list of connectivity matrix (memory-mapped numpy data)
    X : ndarray
        design matrix
    regnames : string array
        regressor names
    nuisance : bool array
        whether a variable is nuisance (True) nor not (False)
    contrast : dictionary of array
        dictionary of contrast vectors.
        Only positive values (sum of multiple effects, a.k.a F-contrast) is
        supported.
    permnum : int
        number of permutation
    exchBlk : dict of array
        'within_block': list of block indice for permuting items within the
                        same index. e.g. [0, 1, 2, 3, 4, 5] with
                        exchBlk['within_block'][1, 1, 2, 2, 3, 3] will be
                        permuted like [1,0,3,2,5,4]
        'whole_block': list of block indice for permuting blocks with the
                        same index as a whole. e.g. [0, 1, 2, 3, 4, 5] with
                        exchBlk['whole_block'][1, 1, 2, 2, 3, 3] will be
                        permuted like [4, 5, 2, 3, 0, 1]
    metric : string
        distance metric (see scipy.spatial.distance.pdist)
        default is 'euclidean'
    chunk_size : float (0, 1) or int
        Size of chunk.  Reduce this number to save memory usage.
        (0, 1) float: ratio of voxels processed at once.
        int: number of voxels processed at once.
        chunk_size == None (default): all data are processed at once.
    verb : bool
        whether to print progress message

    Returns
    -------
    F, pF, Fperm

    Notes
    -----
    Shehzad et al. 2014 NeuroImage
    """

    X = np.array(X)
    nuisance = np.array(nuisance, dtype=np.bool)

    if len(regnames) == 0:
        # Set names of regressor variables
        for xi in range(X.shape[1]):
            regnames.append('var%d' % xi+1)

    # Check nan in connectivity matrices
    maskrm = _check_nan_connectivity(ConnMts, verb)
    """
    Connectivity within the same voxel is NaN so that the same number of
    connectivity as row size of ConnMts (number of source voxels) could be in
    maskrm.
    """

    # -- Prepare vectorized dependent variable, vG ----------------------------
    if verb:
        print("+++ Making vectorized multivariate distance matrix (%s)"
              % time.ctime())
        sys.stdout.flush()

    # N: samples, V: data points (voxels), D: variable dimension
    N = len(ConnMts)
    cmtx = ConnMts[0]
    if cmtx.ndim == 1:
        V = int(np.ceil(np.sqrt(ConnMts[0].shape[0] * 2)))
        V2 = V
    else:
        V, V2 = cmtx.shape

    N, M = X.shape
    Mr = np.linalg.matrix_rank(X)

    # Make centering matrix Ce
    Ce = np.eye(N)-np.ones([N, N])/N

    # Initialize vectorized centered multivariate distance matrix, vG
    vG = np.ndarray([N*N, V], dtype=np.float32)

    # Process for each chunk_size voxels
    if chunk_size is None or chunk_size > V:
        chunk_size = V
    elif chunk_size > 0 and chunk_size < 1:
        chunk_size = int(np.ceil(chunk_size * V))

    for chi, vst in enumerate(range(0, V, chunk_size)):
        vend = vst+chunk_size
        if vend > V:
            vend = V

        # Loading connectivity maps
        if verb:
            st = datetime.datetime.now()
            if chunk_size != V:
                print(" - chunk %d/%d -" % (chi+1, int(np.ceil(V/chunk_size))))
            print("    Loading connectivity maps for %d:%d/%d voxels ..."
                  % (vst, vend, V))
            sys.stdout.flush()

        vn = vend-vst
        rows = range(vst, vend)
        cols = range(V2)
        if maskrm.ndim == 1:
            # exclude voxels with NaN
            rows = np.setdiff1d(rows, maskrm)
            cols = np.setdiff1d(cols, maskrm)

        if V == V2:
            # all pairwise correlation
            # diagonal will be deleted
            Ychunk = np.zeros([len(ConnMts), len(cols)-1, len(rows)],
                              dtype=np.float32)
        else:
            # small volume MDMR mask is used
            Ychunk = np.zeros([len(ConnMts), len(cols), len(rows)],
                              dtype=np.float32)

        for si, cmtx in enumerate(ConnMts):
            if verb:
                print("\r    %d/%d" % (si+1, N), end='')
                sys.stdout.flush()

            if chunk_size == V:
                # Read all
                if cmtx.ndim == 1:
                    connMtx = squareform(cmtx)
                else:
                    connMtx = cmtx.copy()
                np.isnan(cmtx)

                if maskrm.ndim == 1 and len(maskrm):
                    # Remove voxels with nan
                    connMtx = np.delete(connMtx, maskrm, axis=0)
                    connMtx = np.delete(connMtx, maskrm, axis=1)

                if cmtx.ndim == 1:
                    # Remove diagonal
                    connMtx = np.triu(connMtx)[:, 1:] + \
                        np.tril(connMtx)[:, :-1]

                Ychunk[si, :, :] = connMtx.T
            else:
                # Read partial
                if cmtx.ndim == 1:
                    connMtx = _slice_vectorized_dmtx(cmtx, row=rows, col=cols,
                                                     rmdiag=True)
                else:
                    ixgrid = np.ix_(rows, cols)
                    connMtx = cmtx[ixgrid]

                Ychunk[si, :, :] = connMtx.T

            del connMtx

        if verb:
            et = datetime.datetime.now() - st
            print(' done (took %s)' % str(et).split('.')[0])

        # Multivariate distance matrix
        if verb:
            st = datetime.datetime.now()
            print("    Calculating vectorized distance matrix ...")
            sys.stdout.flush()

        prc_intv = 1
        next_prc = prc_intv
        for ii, vi in enumerate(range(vst, vend)):
            if (vi+1)/float(V)*100 > next_prc and verb:
                print("\r    %.2f%%" % ((vi+1)/float(V)*100), end="")
                sys.stdout.flush()
                next_prc += prc_intv
                time.sleep(1)

            if maskrm.ndim == 1:
                D = squareform(pdist(Ychunk[:, :, ii], metric))
            else:
                ytmp = Ychunk[:, :, ii]
                delvi = maskrm[np.nonzero(maskrm[:, 0] == vi)[0], 1]
                ytmp = np.delete(ytmp, delvi, axis=1)
                D = squareform(pdist(ytmp, metric))

            # Squared and halved negative distance matrix (A)
            A = -np.square(D)/2
            # centering A
            G = np.dot(np.dot(Ce, A), Ce)
            # Vectrize G
            vG[:, vi] = G.flatten(order='F')  # vG should be column order ('F')

        del Ychunk
        if verb:
            et = datetime.datetime.now()-st
            print("\r    %.2f%% done (took %s)"
                  % ((vi+1)/float(V)*100, str(et).split('.')[0]))
            sys.stdout.flush()

    # -- Prepare permutation test and set real value as the first permutation -
    if verb:
        print("+++ MDMR with permutation test (%s) ..." % time.ctime())
        sys.stdout.flush()

    """
    H = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)  # Hat matrix
    # H is Hat matrix: H*Y gives Y^, Y estimate with leaset square error
    """
    # Use SVD for rank-deficient X
    U, S, Vt = svd(X, full_matrices=False)
    H = np.dot(U, U.T)

    # Prepare permuted H
    vHp = {}  # vectorized H for permuted samples (1st sample is true sample)
    sortedLabels = []

    # -- Initialize vectorized Hat, Residual, and G matrices with true sample -
    vRp = np.ndarray([permnum, H.size])  # vectorized residual (1-H) matrix
    vRp[0, :] = (np.eye(N)-H).flatten()  # I-H

    # Full F
    lab = 'Full_Fstat'
    sortedLabels.append(lab)
    vHp[lab] = np.ndarray([permnum, H.size])  # vectorized hat matrix
    vHp[lab][0, :] = H.flatten()

    # Sum of effect of interest
    if len(nuisance) - np.sum(nuisance) > 1:
        lab = 'Effects_of_Interest'
        sortedLabels.append(lab)
        vHp[lab] = np.ndarray([permnum, H.size])  # vectorized hat matrix
        Xn = X[:, np.nonzero(nuisance)[0]]
        U, S, Vt = svd(Xn, full_matrices=False)
        Hn = np.dot(U, U.T)
        H2 = H-Hn
        vHp[lab][0, :] = H2.flatten()

    # Individual effect of interest
    for sigi in np.setdiff1d(range(M), np.nonzero(nuisance)[0]):
        lab = regnames[sigi]
        sortedLabels.append(lab)
        vHp[lab] = np.ndarray([permnum, H.size])
        Xn = X[:, np.setdiff1d(range(M), sigi)]
        U, S, Vt = svd(Xn, full_matrices=False)
        Hn = np.dot(U, U.T)
        H2 = H-Hn
        vHp[lab][0, :] = H2.flatten()

    # Contrast
    if len(contrast):
        for lab in sorted(contrast.keys()):
            sortedLabels.append(lab)
            cont = contrast[lab]
            vHp[lab] = np.ndarray([permnum, H.size])
            Xn = X[:, cont == 0]
            U, S, Vt = svd(Xn, full_matrices=False)
            Hn = np.dot(U, U.T)
            H2 = H-Hn
            vHp[lab][0, :] = H2.flatten()

    # -- Make permuted regressor matrices -------------------------------------
    if verb:
        print("    prepare permuted regressors ...", end='')
        sys.stdout.flush()

    # Xn: nuisance regressors
    Xn = X[:, np.nonzero(nuisance)[0]]
    # Xi: effect of interest regressors
    Xi = X[:, np.nonzero(np.logical_not(nuisance))[0]]

    # orthogonalize Xi w.r.t. Xn
    Xnh = np.dot(np.dot(Xn, np.linalg.inv(np.dot(Xn.T, Xn))), Xn.T)
    Xio = np.dot((np.eye(N)-Xnh), Xi)

    # Permutation
    perpat = {}
    for ei in range(Xi.shape[1]):
        perpat[ei] = np.array([','.join([str(v) for v in Xi[:, ei]])])

    for pn in range(1, permnum):
        # Search unique random permutation
        retry = True  # loop flag
        nt = 1  # number of tries
        while retry or nt > permnum:
            if len(exchBlk) == 0:
                # Permute all randomly
                permidx = np.random.permutation(N)
                Xip = Xi[permidx, :]
            else:
                permidx = np.arange(N, dtype=np.int)
                # Permute within exchangeability block
                blist = exchBlk['within_block']
                bidx = np.unique(blist)
                for bi in bidx:
                    p0 = np.nonzero(blist == bi)[0]
                    permidx[p0] = permidx[np.random.permutation(p0)]

                # Permute whole exchangeability block (keep within block order)
                blist = exchBlk['whole_block']
                bidx = np.unique(blist)
                pbidx = np.random.permutation(bidx)  # permuted block id

                permidx1 = np.zeros(len(permidx), dtype=np.int)
                for bi in pbidx:
                    # original positions of block ID bi
                    p0 = np.nonzero(blist == bi)[0]
                    # permuting block ID (bi's position is moved to pbi's
                    # position)
                    pbi = pbidx[bidx == bi][0]
                    # permuted positions of block ID bi
                    p1 = np.nonzero(blist == pbi)[0]
                    permidx1[p1] = permidx[p0]

                assert len(np.unique(permidx1)) == len(permidx1)

                Xip = Xi[permidx1, :]

            retry = False
            for ei in range(Xi.shape[1]):
                if len(np.unique(Xi[:, ei])) == 1:
                    # ignore intercept regressor
                    continue
                pat = [','.join([str(v) for v in Xip[:, ei]])]
                if pat in perpat[ei]:
                    # Same permutation pattern exists
                    retry = True
                    nt += 1
                    break

        if not retry:
            # Save permutation pattern
            for ei in range(Xi.shape[1]):
                pat = [','.join([str(v) for v in Xi[permidx, ei]])]
                perpat[ei] = np.append(perpat[ei], pat)
            if verb:
                print("\r    prepare permuted regressors ... %d/%d" %
                      (pn+1, permnum), end='')
                sys.stdout.flush()
        else:
            break

        # Make permuted design matrix
        Xp = np.ndarray(X.shape, dtype=np.float32)
        Xp[:, np.nonzero(nuisance)[0]] = Xn
        Xp[:, np.nonzero(np.logical_not(nuisance))[0]] = Xio[permidx, :]
        U, S, Vt = svd(Xp, full_matrices=False)
        Hp = np.dot(U, U.T)
        vRp[pn, :] = (np.eye(N)-Hp).flatten()

        # Full
        lab = 'Full_Fstat'
        Xpn = Xp[:, np.nonzero(nuisance)[0]]
        U, S, Vt = svd(Xpn, full_matrices=False)
        Hpn = np.dot(U, U.T)
        vHp[lab][pn, :] = Hpn.flatten()

        # Sum of effect of interest
        if len(nuisance)-np.sum(nuisance) > 1:
            lab = 'Effects_of_Interest'
            Xpn = Xp[:, np.nonzero(nuisance)[0]]
            U, S, Vt = svd(Xpn, full_matrices=False)
            Hpn = np.dot(U, U.T)
            Hp2 = Hp-Hpn
            vHp[lab][pn, :] = Hp2.flatten()

        # Individual effect
        for sigi in np.setdiff1d(range(M), np.nonzero(nuisance)[0]):
            lab = regnames[sigi]
            Xpn = Xp[:, np.setdiff1d(range(M), sigi)]
            U, S, Vt = svd(Xpn, full_matrices=False)
            Hpn = np.dot(U, U.T)
            Hp2 = Hp-Hpn
            vHp[lab][pn, :] = Hp2.flatten()

        # Contrast
        if len(contrast):
            for lab in sorted(contrast.keys()):
                cont = contrast[lab]
                Xpn = Xp[:, cont == 0]
                U, S, Vt = svd(Xpn, full_matrices=False)
                Hpn = np.dot(U, U.T)
                Hp2 = Hp-Hpn
                vHp[lab][pn, :] = Hp2.flatten()

    if not retry:
        if verb:
            print("\r    prepare permuted regressors ... %d/%d done" %
                  (pn+1, permnum))
            sys.stdout.flush()
    else:
        errmsg = "\nMore than %d unique permutation cannot be found.\n" % pn
        errmsg += "\nPermutation number is reduced to %s\n" % (pn-1,)
        sys.stderr.write(errmsg)

        vRp = vRp[:pn, :]
        # Sum of effects of interest
        if len(nuisance)-np.sum(nuisance) > 1:
            lab = 'Effects_of_Interest'
            vHp[lab] = vHp[lab][:pn, :]

        # Individual effect
        for sigi in np.setdiff1d(range(M), np.nonzero(nuisance)[0]):
            lab = regnames[sigi]
            vHp[lab] = vHp[lab][:pn, :]

        # Contrast
        if len(contrast):
            for lab in sorted(contrast.keys()):
                vHp[lab] = vHp[lab][:pn, :]

    # -- Get F and p values ---------------------------------------------------
    if verb:
        print("    evaluate F and p values ...")
        sys.stdout.flush()

    F = {}
    pF = {}
    Fperm = {}
    Deno = np.dot(vRp, vG)
    delvox = np.nonzero(Deno[0, :] == 0)

    # Full_Fstat
    lab = 'Full_Fstat'
    N2 = np.dot(vHp[lab], vG)
    Fs = (N2/Mr)/(Deno/(N-Mr))
    Fs[:, delvox] = 0.0

    F[lab] = Fs[0, :]
    pF[lab] = np.ones(len(F[lab]))
    Fperm[lab] = Fs
    for vn in range(len(pF[lab])):
        pF[lab][vn] = np.sum(Fperm[lab][:, vn] >= F[lab][vn])/float(permnum)

    # Sum of effects of interest
    if len(nuisance)-np.sum(nuisance) > 1:
        lab = 'Effects_of_Interest'
        N2 = np.dot(vHp[lab], vG)
        m2 = np.linalg.matrix_rank(X[:, np.logical_not(nuisance)])
        Fs = (N2/m2)/(Deno/(N-Mr))
        Fs[:, delvox] = 0.0

        F[lab] = Fs[0, :]
        pF[lab] = np.ones(len(F[lab]))
        Fperm[lab] = Fs
        for vn in range(len(pF[lab])):
            pF[lab][vn] = \
                np.sum(Fperm[lab][:, vn] >= F[lab][vn])/float(permnum)

    # Individual regressor
    for sigi in np.setdiff1d(range(M), np.nonzero(nuisance)[0]):
        lab = regnames[sigi]
        N2 = np.dot(vHp[lab], vG)
        Fs = N2/(Deno/(N-Mr))
        Fs[:, delvox] = 0.0

        F[lab] = Fs[0, :]
        pF[lab] = np.ones(len(F[lab]))
        Fperm[lab] = Fs
        for vn in range(len(pF[lab])):
            pF[lab][vn] = \
                np.sum(Fperm[lab][:, vn] >= F[lab][vn])/float(permnum)

    # Contrast
    if len(contrast):
        for lab in sorted(contrast.keys()):
            N2 = np.dot(vHp[lab], vG)
            m2 = np.linalg.matrix_rank(X[:, contrast[lab] != 0])
            Fs = (N2/m2)/(Deno/(N-Mr))
            Fs[:, delvox] = 0.0

            F[lab] = Fs[0, :]
            pF[lab] = np.ones(len(F[lab]))
            Fperm[lab] = Fs
            for vn in range(len(pF[lab])):
                pF[lab][vn] = \
                    np.sum(Fperm[lab][:, vn] >= F[lab][vn])/float(permnum)

    # -- Return ---------------------------------------------------------------
    # Relabel results
    for li, lab in enumerate(sortedLabels):
        replab = '%02d_%s' % (li, lab)
        for v in (F, pF, Fperm):
            v[replab] = v[lab]
            del v[lab]

    if verb:
        print(" done (%s)" % time.ctime())
        sys.stdout.flush()

    return F, pF, Fperm, maskrm


# %% ##########################################################################
def save_map_volume(outfname, F, pF, maskV, aff, pthrs=[0.005, 0.001]):

    labs = sorted(F.keys())
    FV = np.zeros(list(maskV.shape)+[len(F)])
    for ni in range(len(F)):
        Fn = FV[:, :, :, ni]
        Fn[maskV > 0] = F[labs[ni]]
        FV[:, :, :, ni] = Fn

    # thresholded map
    for ni in range(len(F)):
        # uncorrected p
        for pthr in pthrs:
            Fn = np.zeros(FV.shape[:3])
            Fvals = F[labs[ni]].copy()
            Fvals[pF[labs[ni]] > pthr] = 0
            Fn[maskV > 0] = Fvals
            FV = np.append(FV, Fn[:, :, :, np.newaxis], axis=3)
            labs.append(labs[ni]+'(p<%s)' % str(pthr))

    # Save volume and set volume labels and stat parameters
    nim_out = nib.Nifti1Image(FV, aff)
    nib.save(nim_out, outfname)

    # Set volume labels
    try:
        outfname1 = outfname.replace('.nii', '_afni.nii')
        labs = [re.sub(r'\d\d_', '', ll) for ll in labs]
        cmd = f"3dcopy {outfname} {outfname1}; "
        cmd += '3drefit -fim'
        cmd += " -relabel_all_str '%s'" % ' '.join([ll.replace(' ', '_')
                                                    for ll in labs])
        cmd += " %s" % outfname1
        subprocess.call(cmd, shell=True)
    except Exception:
        pass

    lab_f = Path(outfname).stem.replace('.nii', '') + '_label.txt'
    lab_f = Path(outfname).parent / lab_f
    open(lab_f, 'w').write('\n'.join(labs))


# %% ##########################################################################
def cluster_permutation(F, Fperm_npy, maskV, OutPrefix='./', ananame='', NN=1,
                        pthrs=[0.005, 0.001], athrs=[0.05, 0.01], Nproc=0,
                        verb=True):
    """Calculate cluster size threshold by permutation test"""

    # Prepare multiprocessing
    if Nproc == 0:
        Nproc = int(multiprocessing.cpu_count()/2)

    labs = sorted(F.keys())
    for ni in range(len(labs)):
        gc.collect()
        lab1 = re.sub(r'\d\d_', '', labs[ni])
        ofile = OutPrefix + 'ClusterThrPerm.'
        if len(ananame):
            ofile += ananame + '.'
        ofile += '%s.txt' % lab1

        ofile_pdist = ofile.replace('ClusterThrPerm.', 'Cluster_pdist.')
        ofile_pdist = ofile_pdist.replace('.txt', '.pkl')
        if verb:
            st = datetime.datetime.now()
            dstr = str(st).split('.')[0]
            print("-"*80)
            msgstr = "+++ Cluster size permutation test for "
            if len(ananame):
                msgstr += ananame + ':'
            msgstr += labs[ni]
            print(msgstr, end='')
            print(" (%s)" % dstr)
            sys.stdout.flush()

            print("Loading MDMR permutation results ...", end='')
            sys.stdout.flush()

        Fthr = {}
        Fperm = np.load(Fperm_npy[labs[ni]])
        FperSort = np.sort(Fperm, axis=0)
        for pthr in pthrs:
            Fthr[pthr] = np.percentile(FperSort, (1.0-pthr)*100, axis=0)

        if verb:
            print(" done")

            print("Runnning cluster size prmutation test ...")
            sys.stdout.flush()

        clustSizeDist = {}
        clthresh = {}
        for pthr in pthrs:
            Fthrmap = Fperm-Fthr[pthr]
            Fthrmap[Fthrmap < 0] = 0.0

            mp_pool = multiprocessing.Pool(processes=Nproc)

            Nperm = Fthrmap.shape[0]
            pret = [''] * Nperm
            for pmn in range(Nperm):
                Fmaptmp = np.zeros(maskV.shape)
                Fmaptmp[maskV > 0] = Fthrmap[pmn, :]
                pret[pmn] = mp_pool.apply_async(_count_clustsize,
                                                (Fmaptmp, NN))

            mp_pool.close()  # Close processor pool (no more jobs)
            mp_pool.join()  # Wait for finish

            clustSizeDist[pthr] = []
            for pr in pret:
                if pr.successful():
                    clsts = pr.get()
                    clustSizeDist[pthr].append(max(clsts))

                mp_pool.terminate()

            # Evaluate
            clustSizeDist[pthr] = np.sort(clustSizeDist[pthr])
            clthresh[pthr] = {}
            for alpha in athrs:
                clthresh[pthr][alpha] = np.percentile(clustSizeDist[pthr],
                                                      100.0-alpha*100.0)

        del Fperm

        # Save in file
        wtxt = "# 1-sided thresholding\n"
        wtxt += "# %d times permutation\n" % Nperm
        wtxt += "# CLUSTER SIZE THRESHOLD(pthr,alpha) in Voxels\n"
        wtxt += "# -NN %s  | alpha = Prob(Cluster >= given size)\n" % NN
        wtxt += "#  pthr  | %s\n" % ' '.join(['%.5f' % a for a in athrs])
        wtxt += "# ------ | %s\n" % ' '.join(['-------']*len(athrs))
        for pthr in pthrs:
            cthstr = ' '.join(["%7.0f" % clthresh[pthr][a] for a in athrs])
            wtxt += " %.6f  %s\n" % (pthr, cthstr)

        with open(ofile, 'w') as fd:
            fd.write(wtxt)

        with open(ofile_pdist, 'wb') as fd:
            pickle.dump(clustSizeDist, fd)

        if verb:
            et = datetime.datetime.now()
            dstr = str(et).split('.')[0]
            tt = str(et-st).split('.')[0]
            print("done (%s, took %s)\n" % (dstr, tt))
            sys.stdout.flush()
