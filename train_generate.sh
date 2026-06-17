#!/bin/bash

# PSL-Flow main route
DATASET_NAME="${DATASET_NAME:-AVIID}" bash shell/Physics/train_terb_net_dataset.sh
DATASET_NAME="${DATASET_NAME:-AVIID}" bash shell/VAE/train_psl_vae_dataset.sh
SIT_DATASET="${SIT_DATASET:-AVIID}" bash shell/SiT/train_psl_flow_dataset.sh

# KLVAE->SiT ablation
SIT_DATASET="${SIT_DATASET:-AVIID}" bash shell/SiT/train_generic_sit_dataset.sh
