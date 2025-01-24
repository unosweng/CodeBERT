unzip dataset.zip

cd dataset/javaCorpus/
bash download.sh
python preprocess.py --base_dir=token_completion --output_dir=./
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Code/CodeCompletion-line/dataset/javaCorpus/line_completion/test.json

