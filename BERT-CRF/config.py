import os
import torch

data_dir = os.getcwd() + '/data/argi/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
bert_model = 'pretrained_bert_models/bert-base-chinese/'
roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
distilbert_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
albert_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
model_dir = os.getcwd() + '/experiments/argi/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'
use_model = 'bert'


# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
CRF_rate = 100 # CRF的學習率相對於learning_rate的倍數
weight_decay = 0.01
clip_grad = 5

batch_size = 5
epoch_num = 10
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
