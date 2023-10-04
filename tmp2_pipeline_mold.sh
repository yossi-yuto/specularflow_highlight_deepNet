# GPU setting number ex) 0
DEVICE_NUM=$1
# execute date, ex) 20230912
date=$2
kind=$3
iter=$4
read_date=$5

write_dir=${date}/${kind}
read_dir=${read_date}/${kind}
# CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/plastic_mold_dataset -test_mold_type ${kind} -mode rccl -write_dir ${write_dir}/ex_${iter} --train
# CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/plastic_mold_dataset -test_mold_type ${kind} -mode ssf -write_dir ${write_dir}/ex_${iter} --train 
# CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/plastic_mold_dataset -test_mold_type ${kind} -mode sh -write_dir ${write_dir}/ex_${iter} --train
CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/plastic_mold_dataset -test_mold_type ${kind} -mode refine -write_dir ${write_dir}/ex_${iter} --train --eval  
CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -dataset_path data/plastic_mold_dataset -test_mold_type ${kind} -mode pmd -write_dir ${write_dir}/ex_${iter} --train --eval  

