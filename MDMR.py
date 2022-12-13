# -*- coding: utf-8 -*-
"""
Multivariate Distance Matrix Regression (MDMR)
@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import sys
import time
from datetime import timedelta
import re
import subprocess
import multiprocessing
import pickle
import gc

from tqdm import tqdm
import numpy as np
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
from scipy import ndimage
import nibabel as nib
import rpy2.robjects as robjects


# %% _count_clustsize =========================================================
def _count_clustsize(statmap, NN=1):
    """
    Count cluster size

    Parameters
    ----------
    statmap : 3D array
        Thresholded statistical map to find clusters.
    NN : integer, optional
        Cluster definition code.
        1; faces touch
        2; faces or edges touch
        3; faces or edges or corners touch
        The default is 1.

    Returns
    -------
    cluster_sizes : array
        Cluster size distribution.

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


# %% connectivity_matrix ======================================================
def connectivity_matrix(srcV, maskV=None, SVCmask=None, cast_float32=True):
    """
    Calculate Fisher's z-transformed correlation matrix (upper triangle part).

    Parameters
    ----------
    srcV : 4D array
        Function image time series. Noise components should be regressed out.
    maskV : 3D array, optional
        Connectivity calculation mask. If None, voxels with all 0 time series
        are masked. The default is None.
    SVCmask : 3D array, optional
        Mask of the seed voxels. If None, all voxels within maskV are used as
        a seed.
    cast_float32 : TYPE, optional
        Cast connectivity matrix data to float32 to reduce the size.
        The default is True.

    Returns
    -------
    zconnmtx : 1D array
        Half triangle of Fisher's z-transformed correlation matrix.
        Use scipy.spatial.distance.squareform to reconstruct a full matrix.

    """
    Nt = srcV.shape[3]
    src_flat_all = np.reshape(srcV, [-1, Nt])

    # --- Mask ---
    if maskV is None:
        mask_flat =\
            np.nonzero(np.logical_not(np.all(src_flat_all == 0, axis=1)))[0]
    else:
        if not np.all(maskV.shape == srcV.shape[:3]):
            errmsg = f"Mask dimension {maskV.shape}"
            errmsg += f" mismatch with time-course image {srcV.shape[:3]}"
            assert np.all(maskV.shape == srcV.shape[:3]), errmsg

        mask_flat = maskV.flatten()

    # SVC Mask
    if SVCmask is not None:
        if not np.all(SVCmask.shape == srcV.shape[:3]):
            errmsg = f"SVC mask dimension {SVCmask.shape}"
            errmsg += f" mismatch with time-course image {srcV.shape[:3]}"
            assert np.all(SVCmask.shape == srcV.shape[:3]), errmsg

        SVCmask_flat = SVCmask.flatten()

    # Apply mask
    src_flat = src_flat_all[mask_flat > 0, :]
    if SVCmask is not None:
        SVC_src_flat = src_flat_all[SVCmask_flat > 0, :]

    del src_flat_all

    # --- PPI connectivity ---
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

    return zconnmtx


# %% PPI_connectivity_matrix ==================================================
def PPI_connectivity_matrix(srcV, blkReg, maskV=None, cast_float32=True,
                            verb=True):
    """
    Calculate PPI beta matrix between all voxels within the mask.

    Parameters
    ----------
    srcV : 4D array
        Function image time series. Noise components should be regressed out.
    blkReg : TYPE
        Block regressor. Must be the same length as the srcV.shape[-1].
    maskV : 3D array, optional
        Connectivity calculation mask. If None, voxels with all 0 time series
        are masked. The default is None.
    cast_float32 : TYPE, optional
        Cast connectivity matrix data to float32 to reduce the size.
        The default is True.

    Returns
    -------
    PPI_beta_Mtx : 2D array
        PPI beta connectivity matrix (seed x voxels) between all voxels in the
        mask.

    """

    # Check data length
    assert len(blkReg) == srcV.shape[-1], \
        f"Missmatch data length: srcV {srcV.shape[-1]} !=" + \
        f" {len(blkReg)} blkReg"

    Nt = srcV.shape[-1]
    src_flat_all = np.reshape(srcV, [-1, Nt])

    # --- Mask ---
    if maskV is None:
        mask_flat =\
            np.nonzero(np.logical_not(np.all(src_flat_all == 0, axis=1)))[0]
    else:
        if not np.all(maskV.shape == srcV.shape[:3]):
            errmsg = f"Mask dimension {maskV.shape}"
            errmsg += f" mismatch with time-course image {srcV.shape[:3]}"
            assert np.all(maskV.shape == srcV.shape[:3]), errmsg

        mask_flat = maskV.flatten()

    # Apply mask
    src_flat = src_flat_all[mask_flat > 0, :]
    del src_flat_all

    # --- PPI connectivity ---
    PPI_flat = zscore(src_flat * blkReg[None, :], axis=1)
    seed_flat = zscore(src_flat, axis=1)
    Y = seed_flat.T

    # Orthogonalize seed_flat w.r.t PPI_flat
    orth_b = np.sum(PPI_flat / PPI_flat.shape[1] * seed_flat, axis=1)
    orth_seed_flat = seed_flat - orth_b[:, None] * PPI_flat

    # Making X-hat for each seed
    XhAll = None
    Nvox = seed_flat.shape[0]
    for vi in range(Nvox):
        X = np.concatenate(
            [PPI_flat[vi:vi+1, :], orth_seed_flat[vi:vi+1, :],
             blkReg[None, :], np.ones((1, Nt))], axis=0).T
        Nreg = X.shape[1]
        denom = np.linalg.inv(np.dot(X.T, X))
        Xh = np.dot(denom, X.T)
        if XhAll is None:
            XhAll = np.empty((Nreg * Nvox, Nt))

        XhAll[vi*Nreg:(vi+1)*Nreg, :] = Xh

    # Calculate OLS betas for all seed x voxels
    Ball = np.dot(XhAll, Y)

    # Extract betas for the PPI (first) regressor
    PPI_beta_Mtx = Ball[0::X.shape[1], :]

    if cast_float32:
        PPI_beta_Mtx = PPI_beta_Mtx.astype(np.float32)

    return PPI_beta_Mtx


# %% _slice_vectorized_dmtx ===================================================
def _slice_vectorized_dmtx(vdata, row=None, col=None, mdim=None):
    """
    Get a slice of distance matrix from a vectorized data

    Parameters
    ----------
    vdata : array
        Vectorized distance matrix.
    row : array, optional
        Reading row indices. If None, read all rows. The default is None.
    col : array, optional
        Reading column indices. If None, read all columns. The default is None.
    mdim : int, optional
        Size of the distance matrix. The default is None.

    Raises
    ------
    ValueError
        vdata is not a vectorized matrix of mdim x mdim.

    Returns
    -------
    out_mtx : array
        len(row) x len(col) (or len(row) x (len(col)-1) if rmdiag==True),
        matrix extracted from the distance matrix..

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

    outshape = [len(row), len(col)]

    out_mtx = np.zeros(outshape, dtype=vdata.dtype)
    # ri < ci area
    for iir, ri in enumerate(row):
        stp = int(np.sum(range(mdim-1, mdim-1-ri, -1)))
        widx = np.nonzero(col > ri)[0]
        if len(widx) == 0:
            continue

        rdcols = col[widx]-(ri+1) + stp
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


# %% _check_nan_connectivity ==================================================
def _check_nan_connectivity(ConnMtx):
    """
    Find nan in any of ConnMtx item and return indices

    Parameters
    ----------
    ConnMtx : list of arrays
        List of connectivity matrix array (full or upper-triangle vector).

    Returns
    -------
    delvi : array
        List of nan element index.

    """

    dmask = None  # nan mask
    for cmtx in tqdm(ConnMtx, total=len(ConnMtx),
                     desc='Check NaN in connectivity maps'):

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

    return delvi


# %% run_MDMR =================================================================
def run_MDMR(ConnMtx, X, regnames=[], nuisance=[], contrast={}, permnum=10000,
             exchBlk={}, metric='euclidean', rmdiag=True, chunk_size=None,
             rand_seed=0):
    """
    Multivariate Distance Matrix Regression (MDMR)

    Parameters
    ----------
    ConnMtx : list of array
        List of connectivity matrix.
    X : array
        Regressor matrix.
    regnames : string array, optional
        Regressor names. The default is [].
    nuisance : bool array, optional
        Falg od nuisance variables. The default is [].
    contrast : dictionary of array, optional
        Dictionary of contrast vectors. Only positive values (sum of multiple
        effects; F-contrast) is supported. The default is {}.
    permnum : int, optional
        Number of permutation repeat. The default is 10000.
    exchBlk : dictionar of array, optional
        'within_block': Items with the same index are exchanged.
            e.g. When exchBlk['within_block'] = [1, 1, 2, 2, 3, 3],
            items [0, 1, 2, 3, 4, 5] will be permutated like [1, 0, 3, 2, 5, 4].
        'whole_block': Items with the same index are exchanged as a block.
            e.g. When exchBlk['whole_block'] = [1, 1, 2, 2, 3, 3],
            items [0, 1, 2, 3, 4, 5] will be permutated like [2, 3, 4, 5, 0, 1].
        The default is {} means all items are permutated without a restriction.
    metric : str, optional
        Distance metric (see scipy.spatial.distance.pdist).
        The default is 'euclidean'.
    rmdiag : bool, optional
        Flag to remove (replaced with 0) diagonal elements.
        The default is True.
    chunk_size : float [0, 1] or int, optional
        Size of a chunk.  Reduce this number to save memory usage.
        [0, 1] float: ratio of voxels processed at once.
        int: number of voxels processed at once.
        None : all data are processed at once.
        The default is None.
    rand_seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    F : disct of array
        F-value maps.
    pF : disct of array
        p-value maps.
    Fperm : disct of array
        F-values with permutated regressors.
    maskrm : array
        Masked out voxel indices.

    Notes
    -----
    Shehzad et al. 2014 NeuroImage

    """
    np.random.seed(rand_seed)

    X = np.array(X)
    nuisance = np.array(nuisance, dtype=bool)

    if len(regnames) == 0:
        # Set names of regressor variables
        for xi in range(X.shape[1]):
            regnames.append(f'var{xi+1}')

    # Check nan in connectivity matrices
    maskrm = _check_nan_connectivity(ConnMtx)
    """
    Connectivity within the same voxel is NaN so that the same number of
    connectivity as row size of ConnMtx (number of source voxels) could be in
    maskrm.
    """

    # -- Prepare vectorized dependent variable, vG ----------------------------
    print("+++ Making vectorized multivariate distance matrix ",
          f"({time.ctime()})")
    sys.stdout.flush()

    # N: samples, V: data points (voxels), D: variable dimension
    N = len(ConnMtx)
    cmtx = ConnMtx[0]
    if cmtx.ndim == 1:
        V = int(np.ceil(np.sqrt(ConnMtx[0].shape[0] * 2)))
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
        if chunk_size != V:
            print(f"-- Chunk {chi+1}/{int(np.ceil(V/chunk_size))} --")
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
            Ychunk = np.zeros([len(ConnMtx), len(cols), len(rows)],
                              dtype=np.float32)
        else:
            # small volume MDMR mask is used
            Ychunk = np.zeros([len(ConnMtx), len(cols)-1, len(rows)],
                              dtype=np.float32)

        for si, cmtx in tqdm(enumerate(ConnMtx), total=len(ConnMtx),
                             desc='Loading connectivity maps'):
            if chunk_size == V:
                # Read all
                if cmtx.ndim == 1:
                    connMtx = squareform(cmtx)
                else:
                    connMtx = cmtx.copy()

                if rmdiag and V == V2:
                    # Reaplce diagonal with 0
                    connMtx -= np.diag(np.diag(connMtx))

                if len(maskrm):
                    # Remove voxels with nan
                    if maskrm.ndim == 1:
                        connMtx = np.delete(connMtx, maskrm, axis=0)
                        connMtx = np.delete(connMtx, maskrm, axis=1)
                    else:
                        connMtx_list = []
                        for r, c in maskrm:
                            conn = np.delete(connMtx[r, :], c)
                            connMtx_list.append(conn.reshape([1, -1]))
                        connMtx = np.concatenate(connMtx_list, axis=0)
            else:
                # Read partial
                if cmtx.ndim == 1:
                    connMtx = _slice_vectorized_dmtx(cmtx, row=rows, col=cols)
                else:
                    ixgrid = np.ix_(rows, cols)
                    connMtx = cmtx[ixgrid]

                if rmdiag:
                    # Reaplce diagonal with 0
                    diag_mask = np.concatenate(
                        [(cols == ri)[None, :] for ri in rows], axis=0)
                    connMtx[diag_mask] = 0.0

            Ychunk[si, :, :] = connMtx.T
            del connMtx

        # Multivariate distance matrix
        vidxs = list(range(vst, vend))
        for ii, vi in tqdm(enumerate(vidxs), total=len(vidxs),
                           desc='Calculating vectorized distance matrix'):
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

    # -- Prepare permutation test and set real value as the first permutation -
    print(f"+++ MDMR with permutation test ({time.ctime()}) ...")
    sys.stdout.flush()

    """
    H = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)  # Hat matrix
    # H is Hat matrix: H*Y gives Y^, Y estimate with leaset square error
    """
    # Use SVD for rank-deficient X
    U, S, Vt = svd(X, full_matrices=False)
    H = np.dot(U, U.T)

    # Prepare permutated H
    vHp = {}  # vectorized H for permutated samples (1st sample is true sample)
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

    # -- Make permutated regressor matrices -------------------------------------
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

    for pn in tqdm(range(1, permnum), total=permnum-1,
                   desc='Preparing permutated regressors'):
        # Search unique random permutation
        retry = True  # loop flag
        nt = 1  # number of tries
        MaxTry = 10000
        while retry or nt > MaxTry:
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
                pbidx = np.random.permutation(bidx)  # permutated block id

                permidx1 = np.zeros(len(permidx), dtype=np.int)
                for bi in pbidx:
                    # original positions of block ID bi
                    p0 = np.nonzero(blist == bi)[0]
                    # permuting block ID (bi's position is moved to pbi's
                    # position)
                    pbi = pbidx[bidx == bi][0]
                    # permutated positions of block ID bi
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

            # -- End while loop to find a unique permutation --

        if not retry:
            # Save permutation pattern
            for ei in range(Xi.shape[1]):
                pat = [','.join([str(v) for v in Xi[permidx, ei]])]
                perpat[ei] = np.append(perpat[ei], pat)
        else:
            # Could not find a unique permutation for MaxTry times
            break

        # Make permutated design matrix
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

    if retry:
        errmsg = f"\nMore than {pn} unique permutation cannot be found.\n"
        errmsg += f"\nPermutation number is reduced to {pn-1}\n"
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
    print("\n+++ Evaluate F and p values ...")
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
        replab = f'{li:02d}_{lab}'
        for v in (F, pF, Fperm):
            v[replab] = v[lab]
            del v[lab]

    print(f"Done ({time.ctime()})")
    sys.stdout.flush()

    return F, pF, Fperm, maskrm


# %% save_map_volume ==========================================================
def save_map_volume(outfname, F, pF, maskV, aff, pthrs=[0.005, 0.001], FDRths=None):
    """
    Save statistical maps in a nii.gz file.

    Parameters
    ----------
    outfname : str or Path
        Output filename.
    F : list of 3D array
        List of F-value maps.
    pF : list of 3D array
        List of p-value maps.
    maskV : 3D array
        Mask volume.
    aff : array
        Affine matrix for the saved image.
    pthrs : list of float, optional
        List of p-value thresholds. The default is [0.005, 0.001].
    FDRths : list of float, optional
        List of FDR thresholds. The default is None.
    """

    labs = []
    StatVs = np.zeros([*maskV.shape, 0])
    for lab in F.keys():
        # Stat map
        V = np.zeros(maskV.shape)
        V[maskV > 0] = F[lab]
        StatVs = np.concatenate([StatVs, V[:, :, :, None]], axis=-1)
        labs.append(lab)

        # p-value map
        pV = np.zeros(maskV.shape)
        pV[maskV > 0] = pF[lab]
        StatVs = np.concatenate([StatVs, pV[:, :, :, None]], axis=-1)
        labs.append(lab + '_p-value')
        for pthr in pthrs:
            V = np.zeros(maskV.shape)
            fval = F[lab]
            fval[pF[lab] > pthr] = 0
            V[maskV > 0] = fval
            StatVs = np.concatenate([StatVs, V[:, :, :, None]], axis=-1)
            labs.append(lab + f'(p<{pthr})')

        if FDRths is not None:
            robjects.r.assign('pvals', pF[lab])
            qvals = robjects.r('p.adjust(pvals, method = "BH")')
            for fdrth in FDRths:
                V = np.zeros(maskV.shape)
                fval = F[lab]
                fval[qvals > fdrth] = 0
                V[maskV > 0] = fval
                StatVs = np.concatenate([StatVs, V[:, :, :, None]], axis=-1)
                labs.append(lab + f'(FDRq<{fdrth})')

    # Save volume and set volume labels and stat parameters
    nim_out = nib.Nifti1Image(StatVs.astype(np.float32), aff)
    nib.save(nim_out, outfname)

    # Set volume labels
    outfname = Path(outfname)
    try:
        outfname1 = outfname.parent / \
            outfname.name.replace('.nii', '_afni.nii')
        labs = [re.sub(r'\d\d_', '', ll) for ll in labs]
        cmd = f"3dcopy -overwrite {outfname} {outfname1}; "
        subprocess.call(cmd, shell=True)
        cmd = '3drefit -fim'
        labels = ' '.join([ll.replace(' ', '_') for ll in labs])
        cmd += f" -relabel_all_str '{labels}' {outfname1}"
        subprocess.call(cmd, shell=True)
    except Exception:
        pass

    lab_f = outfname.stem.replace('.nii', '') + '_label.txt'
    lab_f = outfname.parent / lab_f
    open(lab_f, 'w').write('\n'.join(labs))


# %% cluster_permutation ======================================================
def cluster_permutation(F, Fperm_npy, maskV, OutPrefix='./', ananame='', NN=1,
                        pthrs=[0.005, 0.001], athrs=[0.05, 0.01], Nproc=0):
    """
    Calculate cluster size threshold by permutation test, and save the result
    statistics in a file.

    Parameters
    ----------
    F : TYPE
        DESCRIPTION.
    Fperm_npy : TYPE
        DESCRIPTION.
    maskV : TYPE
        DESCRIPTION.
    OutPrefix : TYPE, optional
        DESCRIPTION. The default is './'.
    ananame : TYPE, optional
        DESCRIPTION. The default is ''.
    NN : TYPE, optional
        DESCRIPTION. The default is 1.
    pthrs : TYPE, optional
        DESCRIPTION. The default is [0.005, 0.001].
    athrs : TYPE, optional
        DESCRIPTION. The default is [0.05, 0.01].
    Nproc : TYPE, optional
        DESCRIPTION. The default is 0.
    verb : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
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
        ofile += f'{lab1}.txt'

        ofile_pdist = ofile.replace('ClusterThrPerm.', 'Cluster_pdist.')
        ofile_pdist = ofile_pdist.replace('.txt', '.pkl')

        st = time.time()
        print("-"*80)
        msgstr = "+++ Cluster size permutation test for "
        if len(ananame):
            msgstr += ananame + ':'
        msgstr += labs[ni] + f" ({time.ctime(st)})"
        print(msgstr)
        sys.stdout.flush()

        print("Loading MDMR permutation results ...")
        sys.stdout.flush()

        Fthr = {}
        Fperm = np.load(Fperm_npy[labs[ni]])
        FperSort = np.sort(Fperm, axis=0)
        for pthr in pthrs:
            Fthr[pthr] = np.percentile(FperSort, (1.0-pthr)*100, axis=0)

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
        wtxt += f"# {Nperm} times permutation\n"
        wtxt += "# CLUSTER SIZE THRESHOLD (pthr, alpha) in Voxels\n"
        wtxt += f"# -NN {NN}  | alpha = Prob(Cluster >= given size)\n"
        pthstr = ' '.join([f'{a:.5f}' for a in athrs])
        wtxt += f"#  pthr  | {pthstr}\n"
        athstr = ' '.join(['-------']*len(athrs))
        wtxt += f"# ------ | {athstr}\n"
        for pthr in pthrs:
            cthstr = ' '.join([f"{clthresh[pthr][a]:7.0f}" for a in athrs])
            wtxt += f" {pthr:.6f}  {cthstr}\n"

        with open(ofile, 'w') as fd:
            fd.write(wtxt)

        with open(ofile_pdist, 'wb') as fd:
            pickle.dump(clustSizeDist, fd)

        tt = str(timedelta(seconds=time.time()-st)).split('.')[0]
        print(f"done ({time.ctime()}, took {tt})\n")
        sys.stdout.flush()
