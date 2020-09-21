import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm_notebook as tqdm
import random
import numpy as np
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, RandomSampler
import datetime
from sklearn.metrics import f1_score, accuracy_score
from pytorchtools import EarlyStopping

def train_qa(args, rdrsegmenter, bpe, vocab, model):
    log_file = open(args.path_log_file, 'a')
    
    # load dataset
    print("Start loading dataset ...")
    train_dataset, train_dataloader, test_dataset, test_dataloader = save_and_load_dataset(args, rdrsegmenter, bpe, vocab)
    print("Load train dataset done !!!")
    
    #Total steps
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    # Optimizer and Schedule 
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
    # Train
    print("Training ...\n")
    log_file.write("Num train: {}".format(len(train_dataset)))
    log_file.write("\tNum test: {}".format(len(test_dataset)))
    log_file.write("\tNum Epochs: {}".format(args.num_train_epochs))
    log_file.write("\tGradient Accumulation steps: {}".format(args.gradient_accumulation_steps))
    log_file.write("\tTotal steps: {}\n".format(t_total))
    log_file.write("________________________________________________________________")
    
    global_step = 0
    best_f1_score = 0
    model.zero_grad()
    set_seed(args)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.num_train_epochs):
        full_target = []
        full_predict = []
        tr_loss = 0.0

        epoch_iterator = tqdm(train_dataloader, desc="Training", leave=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
#             print(batch[0])
#             print(batch[1])
#             print(batch[2])
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'label': batch[2]}

            loss, predict, target = model.loss(inputs['input_ids'], inputs['attention_mask'], inputs['label'])
            full_target.extend(target)
            full_predict.extend(predict)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                
                # Print some results
                if (args.save_steps > 0 and global_step % args.save_steps == 0):
                    line_start_logging = "\nEpoch: {}\tStep: {}\tLR: {}\n".format(epoch+1, global_step, round(scheduler.get_last_lr()[0], 6))
                    print(line_start_logging)
                    log_file.write(line_start_logging)
                    
                    # Print and save score
                    f1 = f1_score(full_target, full_predict)
                    accuracy = accuracy_score(full_target, full_predict)
                    
                    output_train = {
                        "loss": round(tr_loss / len(train_dataset), 4),
                        "accuracy": round(accuracy, 4),
                        "f1": round(f1, 4)
                    }
                    
                    line_log_train = "Train result:\tLoss: {}\tAcc: {}\tF1: {}\n".format(output_train['loss'], output_train['accuracy'], output_train['f1'])
                    print(line_log_train)
                    log_file.write(line_log_train)

                    output_test = test_qa(args, model, rdrsegmenter, test_dataset, test_dataloader)
                    line_log_test = "Test result:\tLoss: {}\tAcc: {}\tF1: [[{}]]\n".format(output_test['loss'], output_test['accuracy'], output_test['f1'])
                    print(line_log_test)
                    log_file.write(line_log_test)

                    # Save model checkpoint if have better F1 score
                    if (output_test['f1'] > best_f1_score):
                        best_f1_score = output_test['f1']
                        output_dir = os.path.join(args.output_dir, "Epoch{}_Step{}".format(epoch+1, global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(output_dir)
#                         rdrsegmenter.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        log_file.write("Save checkpoint to {}".format(output_dir))
                    print('________________________________________________________________')
                    log_file.write("\n________________________________________________________________\n")
                    
                    early_stopping(output_test['loss'], model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                    
    log_file.close()
    
def test_qa(args, model, tokenizer, test_dataset, test_dataloader):
    total_loss = 0.0
    full_predict = []
    full_target = []

    for batch in tqdm(test_dataloader, desc="Testing", leave=False):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'label': batch[2]}
            loss, predict, target = model.loss(inputs['input_ids'], inputs['attention_mask'], inputs['label'])
            total_loss += loss.item()
            full_predict.extend(predict)
            full_target.extend(target)

    f1 = f1_score(full_target, full_predict)
    accuracy = accuracy_score(full_target, full_predict)

    output_test = {
        "loss": round(total_loss / len(test_dataset), 4),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4)
    }
    return output_test 
        
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
def save_and_load_dataset(args, rdrsegmenter, bpe, vocab):
    # Save dataset to folder './data_pt/phobert_pt'
    if not os.path.exists('./data_pt/phobert_pt'):
        os.makedirs('./data_pt/phobert_pt')
    train_filename = os.path.splitext(os.path.basename(args.path_train_data))[0]
    test_filename = os.path.splitext(os.path.basename(args.path_test_data))[0]
    train_pt_filepath = os.path.join('./data_pt/phobert_pt', train_filename+'.pt')
    test_pt_filepath = os.path.join('./data_pt/phobert_pt', test_filename+'.pt')
    
    if args.load_data_from_pt:
        train_dataset = torch.load(train_pt_filepath)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
        test_dataset = torch.load(test_pt_filepath)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
    else:
        train_dataset, train_dataloader = load_dataset(args.path_train_data, rdrsegmenter, bpe, vocab, args.max_seq_length, args.max_query_length, args.batch_size)
        torch.save(train_dataset, train_pt_filepath)
        test_dataset, test_dataloader = load_dataset(args.path_test_data, rdrsegmenter, bpe, vocab, args.max_seq_length, args.max_query_length, args.batch_size)
        torch.save(test_dataset, test_pt_filepath)
           
    return train_dataset, train_dataloader, test_dataset, test_dataloader

def load_dataset(path_input_data, rdrsegmenter, bpe, vocab, max_seq_length, max_query_length, batch_size):
    examples = read_example(path_input_data)
    random.shuffle(examples)

    features = convert_examples_to_features(examples, rdrsegmenter, bpe, vocab, max_seq_length, max_query_length)

    # Convert to Tensors and build dataset
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    label = torch.tensor([f.label for f in features], dtype=torch.long)
    
    dataset = TensorDataset(input_ids, input_mask, label)
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return dataset, train_dataloader

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, label=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label = label
        
class Example(object):
    def __init__(self, question, answer, label):
        self.question = question
        self.answer = answer
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        string = ""
        string += "question: %s" % (self.question)
        string += ", answer: %s" % (self.answer)
        string += ", label: %r" % (self.label)
        return string

def read_example(input_data):
    with open(input_data, "r", encoding='utf-8') as f:
        examples = []
        for line in f.readlines():
            try:
                line = line.replace("\n", "")
                text = line.split("\t")

                # text[0] : question
                # text[1] : answer
                # text[2] : label
                question = text[0]
                answer = text[1]
                if text[2] == "true":
                    label = 1
                else:
                    label = 0
                try:
                    example = Example(question=question, answer=answer, label=label)
                    examples.append(example)
                except:
                    print("Error on Example")
            except:
                print(line)
    return examples

def convert_examples_to_features(examples, rdrsegmenter, bpe, vocab, max_seq_length, max_query_length):
    features = []
    for (example_index, example) in enumerate(examples):
        # <s> A </s> </s> B </s>
        question_tokens = rdrsegmenter.tokenize(example.question)
        question = ' '.join(token for sen in question_tokens for token in sen)
        question_subwords = bpe.encode(question)
        question_input_ids = vocab.encode_line(question_subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        
        if len(question_input_ids) > max_query_length:
            question_input_ids = question_input_ids[0:max_query_length]
           
        answer_tokens = rdrsegmenter.tokenize(example.answer)
        answer = ' '.join(token for sen in answer_tokens for token in sen)
        answer_subwords = bpe.encode(answer)
        answer_input_ids = vocab.encode_line(answer_subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        
        # The -4 because of 4 tokens:  <s> </s> </s> </s>
        max_answer_tokens = max_seq_length - len(question_input_ids) - 4
        if len(answer_input_ids) > max_answer_tokens:
            answer_input_ids = answer_input_ids[0:max_answer_tokens]
         
        # <s> token id: 0
        # </s> token id: 2
        input_ids = [0] + question_input_ids + [2,2] + answer_input_ids + [2]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
            
        # pad token id: 1
        while len(input_ids) < max_seq_length:
            input_ids.append(1)
            input_mask.append(0)
            
        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(input_mask) == max_seq_length, "Error with input length {} vs {}".format(len(input_mask), max_seq_length)
        
#         print(question_subwords)
#         print(answer_subwords)
#         print(input_ids)

        features.append(InputFeatures(
            input_ids=input_ids, input_mask=input_mask, label=example.label)
        )
    return features