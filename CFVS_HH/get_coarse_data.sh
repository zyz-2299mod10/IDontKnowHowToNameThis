#! /bin/bash

while getopts ":o:t:c:" opt; do
    case "$opt" in
        o)
            object="$OPTARG"
            ;;
        t)
            total_env="$OPTARG"
            ;;
        c)
            chunk_env="$OPTARG"
            ;;        
        \?)
            exit 1
            ;;
    esac
done

# rm -rf ./PDM_dataset/"$object"_coarse/*

chunk_num=$(($total_env / $chunk_env))
for ((i=0;i<chunk_num;i++))
do
    echo "chunk idx " $i 
    python get_coarse_data.py --object $object --num_envs $chunk_env --chunk_idx $i --asset_root /home/hcis/isaacgym/assets --urdf_root /home/hcis/YongZhe/CFVS_HH/obj-and-urdf/urdf
done

python ./depth2pcd.py --visualize --data my --mode train --type coarse --object "$object" 
