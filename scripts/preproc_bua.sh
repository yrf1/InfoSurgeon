DATASET=$1
PROC_DIR_PRISTINE=${PWD}/data/${DATASET}/

if [ ! -d ${PROC_DIR_PRISTINE}/vision/data/jpg/bbox_36 ]
then
mkdir ${PROC_DIR_PRISTINE}/vision/data/jpg/bbox_36
mkdir ${PROC_DIR_PRISTINE}/bottom_up_attention
fi

cd ./data_preproc/bottom-up-attention.pytorch

# Need to download this file from browser
#if [ ! -f bua-caffe-frcn-r101_with_attributes.pth ]
#then
#wget https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EaXvCC3WjtlLvvEfLr3oa8UBLA21tcLh4L8YLbYXl6jgjg?download=1
#fi

cd detectron2
pip install -e .
cd ..

git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
cd ..

python setup.py build develop
pip install ray

python3 extract_features.py --mode caffe \
	  --num-cpus 32 \
	    --extract-mode bboxes \
	      --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml \
	        --image-dir ${PROC_DIR_PRISTINE}/vision/data/jpg/jpg \
		  --out-dir ${PROC_DIR_PRISTINE}/vision/data/jpg/bbox_36 \
		    --resume

python3 extract_features.py --mode caffe \
	  --num-cpus 32 \
	    --extract-mode bbox_feats \
	      --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml \
	        --image-dir ${PROC_DIR_PRISTINE}/vision/data/jpg/jpg \
		  --bbox-dir ${PROC_DIR_PRISTINE}/vision/data/jpg/bbox_36 \
		    --out-dir ${PROC_DIR_PRISTINE}/bottom_up_attention --resume
