export CLASSPATH=${PWD}/NLP_toolbox/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2.jar
DATASET=$3
PROC_DIR_PRISTINE=${PWD}/data/${DATASET}/$2
## Prepare Data Format

## BERT Tokenization
cd data_preproc/PreSumm$1/src/
if [ -d ${PROC_DIR_PRISTINE}/TOKENIZED_PATH ]
then
  rm -r ${PROC_DIR_PRISTINE}/TOKENIZED_PATH
fi
if [ -d ${PROC_DIR_PRISTINE}/JSON_PATH ]
then
  rm -r ${PROC_DIR_PRISTINE}/JSON_PATH
fi
if [ -d ${PROC_DIR_PRISTINE}/BERT_DATA_PATH ]
then
  rm -r ${PROC_DIR_PRISTINE}/BERT_DATA_PATH
fi
if [ -f yi_test.txt ]
then
 rm yi_test.txt
fi

mkdir ${PROC_DIR_PRISTINE}/TOKENIZED_PATH
mkdir ${PROC_DIR_PRISTINE}/JSON_PATH
mkdir ${PROC_DIR_PRISTINE}/BERT_DATA_PATH

for f in ${PROC_DIR_PRISTINE}/txt/*.txt; do
  FILENAME=$(basename ${f%%.*})
  echo "${FILENAME}" >>yi_test.txt
done

python3 preprocess.py -mode tokenize -raw_path ${PROC_DIR_PRISTINE}/txt -save_path ${PROC_DIR_PRISTINE}/TOKENIZED_PATH
python3 preprocess.py -mode format_to_lines -raw_path ${PROC_DIR_PRISTINE}/TOKENIZED_PATH -save_path ${PROC_DIR_PRISTINE}/JSON_PATH/GoodNews -map_path ../urls -lower
python3 preprocess.py -mode format_to_bert -raw_path ${PROC_DIR_PRISTINE}/JSON_PATH -save_path ${PROC_DIR_PRISTINE}/BERT_DATA_PATH -n_cpus 4 -log_file ../logs/preprocess.log
python3 preprocess_collate.py --input_dir ${PROC_DIR_PRISTINE}/BERT_DATA_PATH
