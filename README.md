# Multidimensional Distance Matrix Regression (MDMR) analysis

Please cite studied below when you used this script.

Misaki, M., Phillips, R., Zotev, V., Wong, C.K., Wurfel, B.E., Krueger, F., Feldner, M., Bodurka, J., 2018. Real-time
 fMRI amygdala neurofeedback positive emotional training normalized resting-state functional connectivity in combat veterans with and without PTSD: a connectome-wide investigation. Neuroimage Clin 20, 543-555.

## INSTALL
1. Download the package.

```
git clone https://github.com/mamisaki/MDMR_fMRI.git
```

2. Install python packages.
Here, anaconda is used.
```
conda create -n MDMR python=3.8
conda activate MDMR
conda install tqdm numpy scipy nibabel pandas rpy2 -c conda-forge -c defaults
```

## Example

efer example.py for an example script to run the longitudinal MDMR.
MDMR.py is called from example.py


