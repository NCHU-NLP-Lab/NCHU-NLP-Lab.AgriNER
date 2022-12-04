import os

data_dir = os.getcwd() + '/data/argi/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
vocab_path = data_dir + 'vocab.npz'
exp_dir = os.getcwd() + '/experiments/argi/'
model_dir = exp_dir + 'model.pth'
log_dir = exp_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'

max_vocab_size = 1000000

n_split = 5
dev_split_size = 0.1
# batch_size = 32
batch_size = 2
embedding_size = 128
hidden_size = 384
drop_out = 0.5
lr = 0.001
betas = (0.9, 0.999)
lr_step = 5
lr_gamma = 0.8

# epoch_num = 30
epoch_num = 30
min_epoch_num = 5
patience = 0.0002
patience_num = 5

gpu = ''

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
