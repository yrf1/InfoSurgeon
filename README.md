# InfoSurgeon

Coming soon, thank you for your patience.

### Get the Datasets

First, download and decompress data/[NYTimes_orig](https://uofi.box.com/s/ib5fn1f1h0rbea1z05fwi98rs0481w9l), data/[VOA](https://drive.google.com/file/d/1VAwClg1rlQYFOzZiOEw6gskS7w50VLjL/view?usp=sharing) to be released this week, and [NLP_toolbox](https://uofi.box.com/s/d0ywa0qhlwxtpmrn3n0ye5vd9o4bbj1y). 

### Preprocess Data 

Our pipeline requires several preprocessing steps, from other preexisting works:

* Step 0: _Prepare raw data into a parsed format standard across the two datasets, NYTimes and VOA_

* Step 1: _Bert tokenization for textual summarization features_ [[paper]](https://aclanthology.org/D19-1387.pdf) [[code]](https://github.com/nlpyang/PreSumm)

* Step 2: _Bottom-Up-Attention visual semantic feature extraction_ [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1163.pdf) [[code]](https://github.com/MILVLG/bottom-up-attention.pytorch)

* Step 3: _Building the IE/KG given the news article_ [[paper]](https://aclanthology.org/2020.acl-demos.11/) [[code]](https://github.com/GAIA-AIDA)

We are porting in the code for the above components into our repo so they can be run via the following commands.
```
dataset=NYTimes

## For the first time running the dataset..
if [ "$dataset" == "NYTimes" ]; then
    python scripts/get_NYTimes_data.py  #step 0
fi
sh scripts/preproc_bert.sh "" "" ${dataset}  #step 1a
sh scripts/preproc_bert.sh "" caption ${dataset}  #step 1b
sh scripts/preproc_bert.sh "" title ${dataset}  #step 1c
sh scripts/preproc_bua.sh ${dataset}  #step 2
sh scripts/preproc_IE.sh ${dataset}  #step 3
```

But this is still in beta development phase. If you encounter set-up or runtime issues, please directly check out and run the original preprocessing source code and documentations linked above!

### Run Misinformation Detection
```
python code/engine.py --data_dir data/${dataset} --ckpt_name ${dataset}
```

### Credits & Acknowledgements

The NYTimes dataset orignated from [GoodNews](https://openaccess.thecvf.com/content_CVPR_2019/papers/Biten_Good_News_Everyone_Context_Driven_Entity-Aware_Captioning_for_News_Images_CVPR_2019_paper.pdf), and [Tan et al., 2020](https://cs-people.bu.edu/rxtan/projects/didan/#:~:text=DIDAN%20is%20a%20multimodal%20model,and%20the%20image%20and%20caption) added in multimedia NeuralNews. 

The pristine/unmanipulated VOA news articles used in our data was originally collected by Manling Li. Many thanks to her.

### General Tips

If you would like to view a jupyter notebook running in the remote server from your local machine, do sth along the lines of 
```
jupyter notebook --no-browser --port=5050  # in the server
ssh -N -f -L localhost:5051:localhost:5050 username@server-entry-address  # from local machine
```

