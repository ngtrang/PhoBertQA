from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class BERT(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        
        self.pooler_output = config.pooler_output
        self.second_to_last = config.second_to_last
        self.concat_last_4hl = config.concat_last_4hl
        self.concat_12hl = config.concat_12hl
        self.sum_last_4hl = config.sum_last_4hl
        self.sum_12hl = config.sum_12hl
        
        if self.pooler_output or self.second_to_last or self.sum_last_4hl or self.sum_12hl:
            self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        elif self.concat_last_4hl:
            self.qa_outputs = nn.Linear(config.hidden_size * 4, config.num_labels)
        elif self.concat_12hl:
            self.qa_outputs = nn.Linear(config.hidden_size * 12, config.num_labels)
            
        self.init_weights()

    def compute(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            
        # Only last hidden layer
        if self.pooler_output:
            cls_output = outputs[1]
        # Second to last hidden layer
        elif self.second_to_last:
            cls_output = outputs[2][-1][:,0, ...]
        # Concat last 4 hidden layers
        elif self.concat_last_4hl:
            cls_output = torch.cat((outputs[2][-4][:,0, ...],outputs[2][-3][:,0, ...], outputs[2][-2][:,0, ...], outputs[2][-1][:,0, ...]), -1)
        # Concat 12 hidden layers
        elif self.concat_12hl:
            cls_output = torch.cat((outputs[2][0][:,0, ...],outputs[2][1][:,0, ...], outputs[2][2][:,0, ...], outputs[2][3][:,0, ...],
                                    outputs[2][4][:,0, ...],outputs[2][5][:,0, ...], outputs[2][6][:,0, ...], outputs[2][7][:,0, ...],
                                    outputs[2][8][:,0, ...],outputs[2][9][:,0, ...], outputs[2][10][:,0, ...], outputs[2][11][:,0, ...]), -1)   
        # Sum last 4 hidden layers
        elif self.sum_last_4hl:
            cls_output = torch.stack((outputs[2][-4][:,0, ...],outputs[2][-3][:,0, ...], outputs[2][-2][:,0, ...], outputs[2][-1][:,0, ...])).sum(0)
        # Sum 12 hidden layers
        elif self.sum_12hl:
            cls_output = torch.stack((outputs[2][0][:,0, ...],outputs[2][1][:,0, ...], outputs[2][2][:,0, ...], outputs[2][3][:,0, ...],
                                    outputs[2][4][:,0, ...],outputs[2][5][:,0, ...], outputs[2][6][:,0, ...], outputs[2][7][:,0, ...],
                                    outputs[2][8][:,0, ...],outputs[2][9][:,0, ...], outputs[2][10][:,0, ...], outputs[2][11][:,0, ...])).sum(0)
            
        final_output = self.dropout(cls_output)
        return final_output
            
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        with torch.no_grad():
            final_output = self.compute(input_ids, attention_mask, token_type_ids)
            logits = self.qa_outputs(final_output)
            return logits

    def loss(self, input_ids, attention_mask, token_type_ids, label):
        target = label
        
        final_output = self.compute(input_ids, attention_mask, token_type_ids)
#         print(final_output.size())
        
        logits = self.qa_outputs(final_output)
        loss = F.cross_entropy(logits, target)

        predict_value = torch.max(logits, 1)[1]
        list_predict = predict_value.cpu().numpy().tolist()
        list_target = target.cpu().numpy().tolist()

        return loss, list_predict, list_target