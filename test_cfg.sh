#!/bin/bash

# PSL-Flow evaluation
SIT_DATASET="${SIT_DATASET:-AVIID}" SIT_CKPT="${SIT_CKPT:-auto}" bash shell/SiT/test_psl_flow_dataset.sh

# KLVAE->SiT ablation evaluation
SIT_DATASET="${SIT_DATASET:-AVIID}" SIT_CKPT="${SIT_CKPT:-auto}" bash shell/SiT/test_generic_sit_dataset.sh
