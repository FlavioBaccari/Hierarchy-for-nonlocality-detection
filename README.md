# SDP hierarchy for detecting nonlocality

This repository provides an approximation of the local set of correlations through a semidefinite programing (SDP) relaxation. It is capable of addressing a completely general (N,m,d) scenario, namely involving N parties, m measurement settings with d outcomes per party.
Thanks to its efficient scaling, it is particularly suited for implementation in scenarios with a high number of parties and/or measurements. 
The method can be seen as a relaxation of the usual linear programming techniques and provides all the tools that are usually needed in the study of nonlocal correlations: nonlocality detection, estimation of robustness to white noise and extraction of Bell inequalities. 

The file `local_tools.py` contains the functions related to generating the SDP and analyzing the solution, and a notebook is provided to show some examples.
