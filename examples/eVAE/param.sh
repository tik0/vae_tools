#!/bin/bash

# Last: MMVAE_setup-0b1000_variances_bshared-0.01_buni-0.1_a-0.1_g-0.0_e-200.svg

export EPOCHS="300"
export DIM_LATENT="2"
export USE_AGG="foo"
#BETA_LIST="6 7 8 9 10 11 12 13 14 15 32"
#BETA_LIST="1 2 3 4 5 6 7 8 12 16 32"
# BETA_LIST="2"
# BETA_LIST="8"
# BETA_LIST="4"
BETA_LIST="1 2 4 8 16"
USE_FASHION_LIST="1 0"
for BETA in ${BETA_LIST}; do
  for LABEL in ${LABEL_LIST}; do
    for USE_FASHION in ${USE_FASHION_LIST}; do
        # disable GPU by CUDA_VISIBLE_DEVICES=""
        BETA=${BETA} DIM_LATENT=${DIM_LATENT} LABEL=${LABEL} USE_FASHION=${USE_FASHION} EPOCHS=${EPOCHS} python3 runme.py
    done
  done
done

