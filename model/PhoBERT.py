from transformers import BertPreTrainedModel, RobertaModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class PhoBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(PhoBERT, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        
        self.dropout = nn.Dropout(0.1)
        self.qa_outputs = nn.Linear(config.hidden_size * 4, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        with torch.no_grad():
            outputs = self.roberta(input_ids, attention_mask=attention_mask)
            cls_output = torch.cat((outputs[2][-4][:,0, ...],outputs[2][-3][:,0, ...], outputs[2][-2][:,0, ...], outputs[2][-1][:,0, ...]),-1)
            final_output = self.dropout(cls_output)
            logits = self.qa_outputs(final_output)
            return logits

    def loss(self, input_ids, attention_mask, label):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        cls_output = torch.cat((outputs[2][-4][:,0, ...],outputs[2][-3][:,0, ...], outputs[2][-2][:,0, ...], outputs[2][-1][:,0, ...]),-1)
        final_output = self.dropout(cls_output)
        logits = self.qa_outputs(final_output)
        
        target = label
        loss = F.cross_entropy(logits, target)

        predict_value = torch.max(logits, 1)[1]
        list_predict = predict_value.cpu().numpy().tolist()
        list_target = target.cpu().numpy().tolist()

        return loss, list_predict, list_target