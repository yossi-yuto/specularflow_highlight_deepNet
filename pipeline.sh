# GPU setting number ex) 0
DEVICE_NUM=$1
# execute date, ex) 20230912
date=$2
cd .data/plastic_mold_dataset
KINDS=$(ls .); cd ../..

for kind in ${KINDS}; do
    write_dir=${date}/${kind}
    for iter in {1,2,3}; do
        CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -mode rccl -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} --train
        CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -mode ssf -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} --train 
        CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -mode sh -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} --train
        CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -mode refine -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} --train --eval
        CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python main.py -mode pmd -write_dir ${write_dir}/ex_${iter} -test_mold_type ${kind} --train --eval
        done
done