# Ontology Attention

To improve a BERT-based model's NER performance for clinical texts.

A paper about this project is wating for publishing. We will update the citition information after publication.

## Usage

### Requirements

- pytorch
- huggingface transformers
- scikit-learn
- treelib
- jieba (required by CBLUE)
- gensim (required by CBLUE)
- boto3 (required by CBLUE)

### Preparation

This project reuses some codes from [CBLUE project](git@github.com:CBLUEbenchmark/CBLUE.git). The first time you run cblue_bert.py will download the codes of CBLUE project via git clone.

### Datadownload

#### CMeEE

It can be downloaded at https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414. Extract the dataset into any folder, and modify ```TASK_DATA_DIR``` in ```configs/CMeEE_config.py```. For example. If the data is in ```datasets/CMeEE/```, change the value of ```TASK_DATA_DIR``` by ```TASK_DATA_DIR = "datasets/CMeEE/"```

#### NCBI_disease
It is automatically downloaded by huaggingface datasets in ncbi_bert.py and ncbi_roberta.py

### Reproduce the Results

#### Basic Usage

Run ```cblue_bert.py / ncbi_bert.py / ncbi_robert.py``` by
```
python cblue_bert.py
```

#### Change the Base Model
Modify the variable "model_name" in ```CMeEE_config.py``` for ```cblue_bert.py```, or in ```ncbi_config.py``` for ```ncbi_bert.py```.

#### Change the Metric
The program by default uses the macro-f1 as the metric.
Since most works on CBLUE use micro-f1, for the users who want to use micro-f1 for ```cblue_bert.py```, please use flag ```--metric```. Runing ```cblue_bert.py``` as the follows will use micro-f1
```
python cblue_bert.py --metric micro
```
