datapath=/data2/yoshimura/mirror_detection/DATA/PMD
write_dir=$2

CUDA_VISIBLE_DEVICES=$1 python main_PMD.py -dataset_path $datapath -mode rccl -write_dir ${write_dir} --train -batch_size 32 -patient 50
CUDA_VISIBLE_DEVICES=$1 python main_PMD.py -dataset_path $datapath -mode ssf -write_dir ${write_dir} --train -batch_size 32 -patient 50
CUDA_VISIBLE_DEVICES=$1 python main_PMD.py -dataset_path $datapath -mode sh -write_dir ${write_dir} --train -batch_size 32 -patient 50
CUDA_VISIBLE_DEVICES=$1 python main_PMD.py -dataset_path $datapath -mode refine -write_dir ${write_dir} --train --eval -batch_size 32 -patient 50
