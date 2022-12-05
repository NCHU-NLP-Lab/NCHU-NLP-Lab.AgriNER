import os
import torch

mode = 'test'                          # train/test/both
data_dir = os.getcwd() + '/data/argi/'
train_dir = data_dir + 'train.npz'

# test_dir = data_dir + 'test.npz'
# test_dir = data_dir + 'demo_random_percentage_0.npz'
test_dir = data_dir + 'demo.npz'
# test_dir = data_dir + 'test_random_percentage_50.npz'
files = ['train', 'demo']
is_rePreprocess = True
bert_model = 'pretrained_bert_models/bert-base-chinese/'
# roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
roberta_model = 'pretrained_bert_models/chinese-roberta-wwm-ext/'

# model_dir = os.getcwd() + '/experiments/argi/'
model_dir = os.getcwd() + '/experiments/argi_v8/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'
use_model = 'roberta'
use_rules = False # 是否引入Rules layer
output_to_Console = True # 是否將結果輸出到console，大量文本記得關閉。只在test的時候有效果。

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
CRF_rate = 100                   # CRF的學習率相對於learning_rate的倍數
weight_decay = 0.01
clip_grad = 5

batch_size = 2
epoch_num = 1
min_epoch_num = 5
patience = 0.0002
patience_num = 10

gpu = ''

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

labels = ['Location', 'Plant', 'Chemicals', 'Diseases', 'Technology', 'Climate', 'MC']

label2id = {
    "O": 0,
    "B-Location": 1,
    "B-Plant": 2,
    "B-Chemicals": 3,
    'B-Diseases': 4,
    'B-Technology': 5,
    'B-Climate': 6,
    'B-MC': 7,
    "I-Location": 8,
    "I-Plant": 9,
    "I-Chemicals": 10,
    'I-Diseases': 11,
    'I-Technology': 12,
    'I-Climate': 13,
    'I-MC': 14,
    "S-Location": 15,
    "S-Plant": 16,
    "S-Chemicals": 17,
    'S-Diseases': 18,
    'S-Technology': 19,
    'S-Climate': 20,
    'S-MC': 21,
}

id2label = {_id: _label for _label, _id in list(label2id.items())}
