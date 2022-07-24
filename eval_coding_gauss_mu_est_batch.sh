#!/bin/bash

## Simulation Parameters
n_tbins=1024
n_depths=64
n_mc_samples=1000
# pw_factor_shared=1
pw_factor_shared=10
############ Set noise levels ############ 
min_sbr_exp=-2
max_sbr_exp=0
n_sbr_lvls=27
min_nphotons_exp=2
max_nphotons_exp=4
n_nphotons_lvls=27
# ############ Photon-Starved Regime (supplement) ############
# min_sbr_exp=-2
# max_sbr_exp=1
# n_sbr_lvls=9
# min_nphotons_exp=0
# max_nphotons_exp=2
# n_nphotons_lvls=9

## Create simulation parameter string
SIM_PARAM_STR=' -n_tbins '$n_tbins' -n_depths '$n_depths' -n_mc_samples '$n_mc_samples 
NOISE_PARAM_STR=' -min_max_sbr_exp '$min_sbr_exp' '$max_sbr_exp' -n_sbr_lvls '$n_sbr_lvls
NOISE_PARAM_STR=$NOISE_PARAM_STR' -min_max_nphotons_exp '$min_nphotons_exp' '$max_nphotons_exp' -n_nphotons_lvls '$n_nphotons_lvls
SHARED_PARAM_STR=$SIM_PARAM_STR$NOISE_PARAM_STR' --save_results --account_irf'

echo $SHARED_PARAM_STR

# ############# TruncatedFourier Coding #############
coding_id='TruncatedFourier'
# n_freqs=(2 4 5 8 12 16 20 32 40 64)
n_freqs=(8 16)
pw_factor=$pw_factor_shared
rec_algo_id='ncc'
SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
CURR_CODING_PARAM_STR=''
n_simulations=${#n_freqs[@]}
 
## Use bash for loop 
for (( i=0; i<$n_simulations; i++ )); do
    CURR_CODING_PARAM_STR=' -n_freqs '${n_freqs[$i]}
    echo python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
    python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
done

############# Identity Coding (Matched Filter Rec) #############
coding_id='Identity'
pw_factor=$pw_factor_shared
rec_algo_id='matchfilt'
SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
CURR_CODING_PARAM_STR=''

python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 

############# PSeriesGray Coding #############
coding_id='PSeriesGray'
# n_psergray_codes=(4 8 16 32 64 128)
n_psergray_codes=(8 16)
pw_factor=$pw_factor_shared
rec_algo_id='ncc'
SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
CURR_CODING_PARAM_STR=''
n_simulations=${#n_psergray_codes[@]}
 
## Use bash for loop 
for (( i=0; i<$n_simulations; i++ )); do
    CURR_CODING_PARAM_STR=' -n_psergray_codes '${n_psergray_codes[$i]}
    echo python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
    python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
done

# ############# PSeriesFourier Coding #############
coding_id='PSeriesFourier'
# n_freqs=(2 4 5 8 12 16 20 32 40 64)
n_freqs=(8 16)
pw_factor=$pw_factor_shared
rec_algo_id='ncc'
SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
CURR_CODING_PARAM_STR=''
n_simulations=${#n_freqs[@]}
 
## Use bash for loop 
for (( i=0; i<$n_simulations; i++ )); do
    CURR_CODING_PARAM_STR=' -n_freqs '${n_freqs[$i]}
    echo python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
    python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
done

# ############# Timestamp Coding (matchfilt rec) #############
# coding_id='Timestamp'
# # n_timestamps=(4 8 10 16 24 32 40 64)
# # n_timestamps=(4 8 10 16 24 32)
# # n_timestamps=(32 64 )
# # n_timestamps=(8 16 )
# n_timestamps=(8 128 )
# pw_factor=$pw_factor_shared
# rec_algo_id='matchfilt'
# SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
# CURR_CODING_PARAM_STR=''
# n_simulations=${#n_timestamps[@]}
 
# ## Use bash for loop 
# for (( i=0; i<$n_simulations; i++ )); do
#     CURR_CODING_PARAM_STR=' -n_timestamps '${n_timestamps[$i]}
#     echo python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
#     python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
# done

############# Gated Coding (Narrow Pulse Width - Quantization Limited) #############
coding_id='Gated'
n_gates=(8 16 32 64)
pw_factor=$pw_factor_shared
rec_algo_id='linear'
SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
CURR_CODING_PARAM_STR=''
n_simulations=${#n_gates[@]}
 
## Use bash for loop 
for (( i=0; i<$n_simulations; i++ )); do
    CURR_CODING_PARAM_STR=' -n_gates '${n_gates[$i]}
    echo python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
    python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
done

############# Gated Coding (Wide Pulse Width - Noise Limited) #############
coding_id='Gated'
n_gates=( 8 16 32 64 )
rec_algo_id='linear'
SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
CURR_CODING_PARAM_STR=''
n_simulations=${#n_gates[@]}
 
## Use bash for loop 
for (( i=0; i<$n_simulations; i++ )); do
    pw_factor=$((n_tbins / ${n_gates[$i]}))
    # echo "n_gates: "${n_gates[$i]}" before: "$pw_factor
    # If the pw_factor is smaller, the truncate it
    if [ "$pw_factor" -lt "$pw_factor_shared" ]; then 
        pw_factor=$pw_factor_shared
    fi
    # echo "n_gates: "${n_gates[$i]}" after: "$pw_factor
    SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
    CURR_CODING_PARAM_STR=' -n_gates '${n_gates[$i]}
    echo python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
    python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
done


# ############# GatedFourier Coding (Narrow Pulse Width - Quantization Limited) #############
# coding_id='GatedFourier-F-1'
# n_gates=(4 8 16 32 64)
# pw_factor=$pw_factor_shared
# rec_algo_id='ncc'
# SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
# CURR_CODING_PARAM_STR=''
# n_simulations=${#n_gates[@]}
 
# ## Use bash for loop 
# for (( i=0; i<$n_simulations; i++ )); do
#     CURR_CODING_PARAM_STR=' -n_gates '${n_gates[$i]}
#     echo python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
#     python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
# done

# # ############# RandomFourier Coding #############
# coding_id='RandomFourier'
# n_rand_freqs=(2 4 5 8 10 16 32 64)
# pw_factor=$pw_factor_shared
# rec_algo_id='ncc'
# SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
# CURR_CODING_PARAM_STR=''
# n_simulations=${#n_rand_freqs[@]}
 
# ## Use bash for loop 
# for (( i=0; i<$n_simulations; i++ )); do
#     CURR_CODING_PARAM_STR=' -n_rand_freqs '${n_rand_freqs[$i]}
#     echo python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
#     python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
# done

# ############# Gray Coding #############
# coding_id='Gray'
# n_bits=(4 8 10)
# pw_factor=$pw_factor_shared
# rec_algo_id='ncc'
# SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
# CURR_CODING_PARAM_STR=''
# n_simulations=${#n_bits[@]}
 
# ## Use bash for loop 
# for (( i=0; i<$n_simulations; i++ )); do
#     CURR_CODING_PARAM_STR=' -n_bits '${n_bits[$i]}
#     echo python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
#     python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
# done

# ############# WalshHadamard Coding #############
# coding_id='WalshHadamard'
# n_wh_codes=(4 8 16 32 64 128 256)
# pw_factor=$pw_factor_shared
# rec_algo_id='ncc'
# SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
# CURR_CODING_PARAM_STR=''
# n_simulations=${#n_wh_codes[@]}
 
# ## Use bash for loop 
# for (( i=0; i<$n_simulations; i++ )); do
#     CURR_CODING_PARAM_STR=' -n_wh_codes '${n_wh_codes[$i]}
#     echo python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
#     python eval_coding_gauss_mu_est.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
# done



