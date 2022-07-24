#!/bin/bash

# run eval_coding_flash_lidar_scene.py -scene_id kitchen-2 -sbr 0.5 -nphotons 1000 -n_rows 240 -n_cols 320 -n_tbins 2000 -coding PSeriesFourier -n_freqs 8

## Simulation Parameters
scene_id=kitchen-2 # Options: kitchen-2, bathroom-cycles-2
# scene_id=bathroom-cycles-2 # Options: kitchen-2, bathroom-cycles-2
n_tbins=2000
n_rows=240
n_cols=320
pw_factor_shared=1 # pulse width in units of time bins
# Set noise levels
sbr=1
nphotons=2000

## Set shared coding params
# n_codes=(10 20 40 80)
# n_codes=(3 8 10 20)
# n_codes=(20)
n_codes=(20 40 80)

## Create simulation parameter string
SIM_PARAM_STR=' -scene_id '$scene_id' -n_tbins '$n_tbins' -n_rows '$n_rows' -n_cols '$n_cols 
NOISE_PARAM_STR=' -sbr '$sbr' -nphotons '$nphotons
SHARED_PARAM_STR=$SIM_PARAM_STR$NOISE_PARAM_STR' --save_data_results --account_irf'

echo $SHARED_PARAM_STR

############# TruncatedFourier Coding #############
coding_id='TruncatedFourier'
pw_factor=$pw_factor_shared
rec_algo_id='ncc'
SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
CURR_CODING_PARAM_STR=''
n_simulations=${#n_codes[@]}
## Use bash for loop 
for (( i=0; i<$n_simulations; i++ )); do
    n_freqs=$((n_codes[$i] / 2))
    CURR_CODING_PARAM_STR=' -n_freqs '${n_freqs}
    echo python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
    python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
done

############# PSeriesFourier Coding #############
coding_id='PSeriesFourier'
pw_factor=$pw_factor_shared
rec_algo_id='ncc'
SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
CURR_CODING_PARAM_STR=''
n_simulations=${#n_codes[@]}
## Use bash for loop 
for (( i=0; i<$n_simulations; i++ )); do
    n_freqs=$((n_codes[$i] / 2))
    CURR_CODING_PARAM_STR=' -n_freqs '${n_freqs}
    echo python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
    python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
done

############# PSeriesGray Coding -- Similar to GrayCoding when K <= log2(N)#############
coding_id='PSeriesGray'
pw_factor=$pw_factor_shared
rec_algo_id='ncc'
SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
CURR_CODING_PARAM_STR=''
n_simulations=${#n_codes[@]}
## Use bash for loop 
for (( i=0; i<$n_simulations; i++ )); do
    CURR_CODING_PARAM_STR=' -n_psergray_codes '${n_codes[$i]}
    echo python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
    python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
done

############# Gated Coding (Narrow Pulse Width - Quantization Limited) #############
coding_id='Gated'
pw_factor=$pw_factor_shared
rec_algo_id='linear'
SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
CURR_CODING_PARAM_STR=''
n_simulations=${#n_codes[@]}
 
## Use bash for loop 
for (( i=0; i<$n_simulations; i++ )); do
    CURR_CODING_PARAM_STR=' -n_gates '${n_codes[$i]}
    echo python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
    python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
done

############# Gated Coding (Wide Pulse Width - Quantization Limited) #############
coding_id='Gated'
rec_algo_id='linear'
SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
CURR_CODING_PARAM_STR=''
n_simulations=${#n_codes[@]}
## Use bash for loop 
for (( i=0; i<$n_simulations; i++ )); do
    pw_factor=$((n_tbins / ${n_codes[$i]}))
    # If the pw_factor is smaller, the truncate it
    if [ "$pw_factor" -lt "$pw_factor_shared" ]; then 
        pw_factor=$pw_factor_shared
    fi
    SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
    CURR_CODING_PARAM_STR=' -n_gates '${n_codes[$i]}
    echo python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
    python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
done

############# Identity Coding (Matched Filter Rec) #############
coding_id='Identity'
pw_factor=$pw_factor_shared
rec_algo_id='matchfilt'
SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
CURR_CODING_PARAM_STR=''

python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 

# ############# Random Coding #############
# coding_id='Random'
# pw_factor=$pw_factor_shared
# rec_algo_id='ncc'
# SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
# CURR_CODING_PARAM_STR=''
# n_simulations=${#n_codes[@]}
# ## Use bash for loop 
# for (( i=0; i<$n_simulations; i++ )); do
#     CURR_CODING_PARAM_STR=' -n_random_codes '${n_codes[$i]}
#     echo python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
#     python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
# done

# ############# HighFreqFourier Coding #############
# coding_id='HighFreqFourier'
# pw_factor=$pw_factor_shared
# rec_algo_id='ncc'
# SHARED_CODING_PARAM_STR=' -coding '$coding_id' -pw_factors '$pw_factor' -rec '$rec_algo_id
# CURR_CODING_PARAM_STR=''
# n_simulations=${#n_codes[@]}
# start_high_freq=40
# ## Use bash for loop 
# for (( i=0; i<$n_simulations; i++ )); do
#     n_high_freqs=$((n_codes[$i] / 2))
#     CURR_CODING_PARAM_STR=' -n_high_freqs '${n_high_freqs}' -start_high_freq '${start_high_freq}
#     echo python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR
#     python eval_coding_flash_lidar_scene.py $SHARED_PARAM_STR $SHARED_CODING_PARAM_STR $CURR_CODING_PARAM_STR 
# done
