# Multidimensional Distance Matrix Regression (MDMR) analysis for fMRI
Python library for multidimensional distance matrix regression (MDMR) analysis of fMRI data.  
MDMR enables connectome-wide association analysis that explores the relationship between variables of interest, such as diagnosis groups and symptom scales, and whole-brain voxel-by-voxel functional connectivity without a priori hypothesis. The library also provides longitudinal design models with example scripts.

Please cite the study below when you use this script.

Misaki, M., Phillips, R., Zotev, V., Wong, C.K., Wurfel, B.E., Krueger, F., Feldner, M., Bodurka, J., 2018. Real-time fMRI amygdala neurofeedback positive emotional training normalized resting-state functional connectivity in combat veterans with and without PTSD: a connectome-wide investigation. Neuroimage Clin 20, 543-555.

## INSTALL
1. Download the package  
```
git clone https://github.com/mamisaki/MDMR_fMRI.git
```

2. Install python packages  
Anaconda (https://www.anaconda.com/) is used here.
```
conda create -n MDMR python=3.8
conda activate MDMR
conda install tqdm numpy scipy nibabel pandas rpy2 -c conda-forge -c defaults
```

## Example
Refer example_single.py for an example script to run MDMR.  
Refer example_longitudinal.py for an example script to run the longitudinal MDMR.

### example_single.py
1. Setup  
  The script read the data list from example/DataList.csv file. The list includes preprocessed fMRI files and regressor values used to make a design matrix.  
  - 'MNI152_T1_2mm_brain_mask.nii.gz': The brain mask file in the same space (MNI) as the preprocessed function images. The voxel resolution can be different.
  - 'aseg.nii.gz': Segmentation image made by FreeSurfer for the MNI template brain.

2. Mask and downsampling  
  AFNI (https://afni.nimh.nih.gov/) is required for image processing.  
  The process includes making the gray matter mask, resampling the mask to the function image resolution, downsampling function images, and downsampling the gray matter mask.
  MDMR analysis mask is the downsampled gray matter mask excluding the voxels with no signal variance.

3. Make connectivity matrices from downsampled images  
  Connectivity matrix is calculated and saved in a numpy binary file. The data is loaded as a memory-mapped numpy array.

4. Make the design matrix  
  R functions are used to make the design matrix, X.  
  Edit the model equation according to the analysis.  
  ```
  # --- Set model equation ---
  model = 'Diagnosis+Sex+Age+Motion'
  ```
   
  - Set the nuisance variables  
    Variables other than the nuisance variables are considered as the effect of interest.  
    In the permutation test, nuisance variables are fixed, and the variables of interest effect are permuted (Winkler et al., 2014).  
      
    Winkler, A.M., Ridgway, G.R., Webster, M.A., Smith, S.M. & Nichols, T.E. Permutation inference for the general linear model. Neuroimage 92, 381-397 (2014).  
      
  - Set exchangeability block: dictionary   
    The exchangeability block defines the blocks of permutation patterns (Winkler et al., 2015).
    'within_block': Items with the same index are exchanged.  
      e.g. When exchBlk['within_block'] = [1, 1, 2, 2, 3, 3], items [0, 1, 2, 3, 4, 5] will be permuted like [1, 0, 3, 2, 5, 4].  
    'whole_block': Items with the same index are exchanged as a block.  
      e.g. When exchBlk['whole_block'] = [1, 1, 2, 2, 3, 3], items [0, 1, 2, 3, 4, 5] will be permuted like [2, 3, 4, 5, 0, 1].  
            
    When no exchangeability block is set, permutation randomly exchanges the samples without a restriction.  
      
    Winkler, A.M., Webster, M.A., Vidaurre, D., Nichols, T.E. & Smith, S.M. Multi-level block permutation. Neuroimage 123, 253-268 (2015).  

5. Run MDMR  
  Run the MDMR analysis and save the results in a pickle file and NIfTI image.
  - Set the parameters of MDMR.  
    - metric: metric of distance matrix between the connectivity maps. Any metrics defined in scipy.spatial.distance.pdist can be used. Default is 'euclidian' distance.  
    - chunk_size: Number of voxels processed at once. Reduce this number if available memory space is limited. If chunk_size == None (default), all data are processed at once.

6. Cluster size permutation test  
  Permutation test for evaluating the cluster-size correction threshold.  
  The results will be saved in example/stats/*ClusterThrPerm.[contrastname].txt file.
  
