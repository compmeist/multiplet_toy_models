# multiplet_toy_models
Early exploratory quick investigations of models for multiplet mlp networks.  Do not use.  These are experimental and cluttered Python programs, as outcomes of preprint An Informal Introduction to Multiplet Neural Networks on arxiv.

The incorporation (rolling in) of selection weights w_k into powered terms is explored (in mm_cs_wk_2c.py), by normalizing the magnitude of the W_k terms (requiring the selection weights to be max-norm 1).   This allows the denominator term to be ignored, but derivatives are reformulated for backprop using this mod.


-Nathan
