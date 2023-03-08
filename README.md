# Phonon workflows

Just some scripts for playing around with phonons

`tblite_phonons.py` performs geometry optimisation and finite-displacement calculations using tblite, ASE and phonopy. To make this work I install tblite-python and phonopy from conda-forge using mamba. It outputs a *phonopy_params.yaml* file; running `phonopy --include-all` in the same directory generates a `phonopy.yaml` file compatible with AbINS and Euphonic.
