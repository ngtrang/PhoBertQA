from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class BERT_LSTM(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT_LSTM, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.bert_dropout = nn.Dropout(0.1)

        self.hidden_size =  config.lstm_hidden_size
        self.num_layers = config.lstm_num_layers
        self.lstm_dropout = config.lstm_dropout
        self.bidirectional = config.bidirectional
        
        if config.bidirectional:
            self.lstm = nn.LSTM(input_size=config.hidden_size * 4, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.lstm_dropout, bidirectional=True)
            self.qa_outputs = nn.Linear(self.hidden_size*2, config.num_labels)
        else:
            self.lstm = nn.LSTM(input_size=config.hidden_size * 4, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.lstm_dropout, bidirectional=False)
            self.qa_outputs = nn.Linear(self.hidden_size, config.num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # Concat last 4 hidden layers
            cls_output = torch.cat((outputs[2][-4][:,0, ...],outputs[2][-3][:,0, ...], outputs[2][-2][:,0, ...], outputs[2][-1][:,0, ...]),-1)

            cls_output = self.lstm(cls_output.unsqueeze(0))[0]
            logits = self.qa_outputs(cls_output)[0]
            return logits

    def loss(self, input_ids, attention_mask, token_type_ids, label):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Concat last 4 hidden layers
        cls_output = torch.cat((outputs[2][-4][:,0, ...],outputs[2][-3][:,0, ...], outputs[2][-2][:,0, ...], outputs[2][-1][:,0, ...]),-1)
        cls_output = self.lstm(cls_output.unsqueeze(0))[0]
        logits = self.qa_outputs(cls_output)[0]
        
        target = label
        
        loss = F.cross_entropy(logits, target)

        predict_value = torch.max(logits, 1)[1]
        list_predict = predict_value.cpu().numpy().tolist()
        list_target = target.cpu().numpy().tolist()

        return loss, list_predict, list_target