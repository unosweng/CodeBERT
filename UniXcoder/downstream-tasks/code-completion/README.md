## Running `run.py` with TRAIN mode

```
Every 2.0s: nvidia-smi                            OISSE-IST173C01: Fri Jan 24 16:11:06 2025

Fri Jan 24 16:11:06 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.142                Driver Version: 550.142        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 6000 Ada Gene...    On  |   00000000:C1:00.0 Off |                  Off |
| 56%   83C    P2            289W /  300W |   48611MiB /  49140MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA RTX 6000 Ada Gene...    On  |   00000000:E1:00.0 Off |                  Off |
| 56%   83C    P2            283W /  300W |   45828MiB /  49140MiB |     93%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      3040      G   /usr/lib/xorg/Xorg                              4MiB |
|    0   N/A  N/A    185348      C   python                                      48592MiB |
|    1   N/A  N/A      3040      G   /usr/lib/xorg/Xorg                             10MiB |
|    1   N/A  N/A      3133      G   /usr/bin/gnome-shell                            8MiB |
|    1   N/A  N/A    185348      C   python                                      45792MiB |
+-----------------------------------------------------------------------------------------+
```


---

# Code Completion

## Dependency 

- pip install torch
- pip install transformers
- pip install javalang

## Data Download

```bash
unzip dataset.zip

cd dataset/javaCorpus/
bash download.sh
python preprocess.py --base_dir=token_completion --output_dir=./
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Code/CodeCompletion-line/dataset/javaCorpus/line_completion/test.json

cd ../py150
bash download.sh
python preprocess.py --base_dir=py150_files --output_dir=./
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Code/CodeCompletion-line/dataset/py150/line_completion/test.json

cd ../..
```



## Fine-Tune Setting

Here we provide fine-tune settings for code completion, whose results are reported in the paper.

#### JavaCorpus Dataset

```shell
# Training
python run.py \
	--do_train \
	--do_eval \
	--lang java \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/javaCorpus/train.txt \
	--dev_filename dataset/javaCorpus/dev.json \
  --output_dir saved_models/javaCorpus \
  --max_source_length 936 \
  --max_target_length 64 \
  --beam_size 5 \
  --train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10
  
# Output predictions of test set
python run.py \
	--do_test \
	--lang java \
	--model_name_or_path microsoft/unixcoder-base \
	--load_model_path saved_models/javaCorpus/checkpoint-best-acc/pytorch_model.bin \
	--test_filename dataset/javaCorpus/test.json \
  --output_dir saved_models/javaCorpus \
  --max_source_length 936 \
  --max_target_length 64 \
  --beam_size 5 \
  --eval_batch_size 32
```

Prediction results of test set are  ```saved_models/javaCorpus/predictions.txt```.To obtain the score of test set, you need to send the prediction to codexglue@microsoft.com.


#### PY150 Dataset

```shell
# Training
python run.py \
	--do_train \
	--do_eval \
	--lang python \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/py150/train.txt \
	--dev_filename dataset/py150/dev.json \
  --output_dir saved_models/py150 \
  --max_source_length 936 \
  --max_target_length 64 \
  --beam_size 5 \
  --train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 32 \
  --learning_rate 2e-4 \
  --num_train_epochs 10
  
# Output predictions of test set  
python run.py \
	--do_test \
	--lang python \
	--model_name_or_path microsoft/unixcoder-base \
	--load_model_path saved_models/py150/checkpoint-best-acc/pytorch_model.bin \
	--test_filename dataset/py150/test.json \
  --output_dir saved_models/py150 \
  --max_source_length 936 \
  --max_target_length 64 \
  --beam_size 5 \
  --eval_batch_size 32
```

Prediction results of test set are  ```saved_models/py150/predictions.txt```.To obtain the score of test set, you need to send the prediction to codexglue@microsoft.com.


