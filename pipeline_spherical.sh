# GPU setting number ex) 0
DEVICE_NUM=$1
# execute date, ex) 20230912
date=$2
iter=$3
write_dir=${date}

CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/spherical_mirror_dataset -mode refine -write_dir ${write_dir}/ex_${iter} --eval
# for iter in {1,2,3}; do
#     CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/spherical_mirror_dataset -mode rccl -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} --train
#     CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/spherical_mirror_dataset -mode ssf -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} --train 
#     CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/spherical_mirror_dataset -mode sh -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} --train
#     CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/spherical_mirror_dataset -mode refine -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} --eval
#     CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/spherical_mirror_dataset -mode pmd -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} --eval
# done
    