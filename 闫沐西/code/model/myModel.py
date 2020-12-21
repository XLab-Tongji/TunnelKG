import sys, json
import torch
import os
import numpy as np
import opennre
from opennre import encoder, model, framework
import sys
import os
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='',
                    help='Checkpoint name')
parser.add_argument('--only_test', action='store_true',
                    help='Only run test')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                    help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='sanwenchar', choices=['none', 'wiki_distant', 'nyt10'],
                    help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='../sanwenc/train.txt', type=str,
                    help='Training data file')
parser.add_argument('--val_file', default='../sanwenc/valid.txt', type=str,
                    help='Validation data file')
parser.add_argument('--test_file', default='../sanwenc/test.txt', type=str,
                    help='Test data file')
parser.add_argument('--rel2id_file', default='../sanwenc/rel2id.json', type=str,
                    help='Relation to ID file')

# Bag related
parser.add_argument('--bag_size', type=int, default=0,
                    help='Fixed bag size. If set to 0, use original bag sizes')

# Hyper-parameters
parser.add_argument('--batch_size', default=160, type=int,
                    help='Batch size')
parser.add_argument('--lr', default=0.1, type=float,
                    help='Learning rate')
parser.add_argument('--optim', default='sgd', type=str,
                    help='Optimizer')
parser.add_argument('--weight_decay', default=1e-5, type=float,
                    help='Weight decay')
parser.add_argument('--max_length', default=120, type=int,
                    help='Maximum sentence length')
parser.add_argument('--max_epoch', default=100, type=int,
                    help='Max number of training epochs')

args = parser.parse_args()

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
if len(args.ckpt) == 0:
    args.ckpt = '{}_{}'.format(args.dataset, 'cnn_soft')
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

root_path = "../sanwenc"

rel2id = json.load(open(args.rel2id_file))

word2id = json.load(open(os.path.join(root_path, 'word2id.json')), )
word2vec = np.load(os.path.join(root_path, 'myvec.npy'))

# Define the sentence encoder
sentence_encoder = opennre.encoder.CNNEncoder(
    token2id=word2id,
    max_length=args.max_length,
    word_size=100,
    position_size=10,
    hidden_size=230,
    blank_padding=True,
    kernel_size=3,
    padding_size=1,
    word2vec=word2vec,
    dropout=0.5
)

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=100,
    lr=args.lr,
    weight_decay=args.weight_decay,
    opt='sgd'
)

# Train the model
if not args.only_test:
    framework.train_model(args.metric)

# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
logging.info('Test set results:')
if args.metric == 'acc':
    logging.info('Accuracy: {}'.format(result['acc']))
else:
    logging.info('Micro precision: {}'.format(result['micro_p']))
    logging.info('Micro recall: {}'.format(result['micro_r']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))
