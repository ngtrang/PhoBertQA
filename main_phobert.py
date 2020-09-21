from transformers import RobertaTokenizer, RobertaConfig
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from vncorenlp import VnCoreNLP
from model.PhoBERT import PhoBERT
from train_phobert import *
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--folder_model', type=str, default='./PhoBERT_base_transformers/model.bin')
parser.add_argument('--config_path', type=str, default='./PhoBERT_base_transformers/config.json')
parser.add_argument('--dict_path', type=str, default='./PhoBERT_base_transformers/dict.txt')
parser.add_argument('--bpe_codes', type=str, default='./PhoBERT_base_transformers/bpe.codes')
parser.add_argument('--rdrsegmenter_path', type=str, default='./VnCoreNLP/VnCoreNLP-1.1.1.jar')
parser.add_argument('--path_train_data', type=str)
parser.add_argument('--path_test_data', type=str)
parser.add_argument('--load_data_from_pt', action='store_true')
parser.add_argument('--path_log_file', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--max_seq_length', type=int, default=256)
parser.add_argument('--max_query_length', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--num_labels', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=3e-5)
parser.add_argument('--gradient_accumulation_steps', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--output_hidden_states', type=bool, default=True)
parser.add_argument('--num_train_epochs', type=int, default=5)
parser.add_argument('--save_steps', type=int, default=60)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=27)
parser.add_argument('--patience', type=int, default=20)

args = parser.parse_args()
bpe = fastBPE(args)

# need remake config with device option for train with another cuda device
config = RobertaConfig.from_pretrained(args.config_path)
config = config.to_dict()
config.update({"device": args.device})
config.update({"output_hidden_states": args.output_hidden_states})
config = RobertaConfig.from_dict(config)

rdrsegmenter = VnCoreNLP(args.rdrsegmenter_path, annotators='wseg')

vocab = Dictionary()
vocab.add_from_file(args.dict_path)

model = PhoBERT.from_pretrained(args.folder_model, config=config)
model = model.to(args.device)
train_qa(args, rdrsegmenter, bpe, vocab, model)