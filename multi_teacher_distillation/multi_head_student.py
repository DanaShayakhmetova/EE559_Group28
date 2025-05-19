from transformers import BertTokenizer, BertModel
import torch 

class MultiHeadStudent(torch.nn.Module):
    """
    Multi-head student model for knowledge distillation using two homogeneous teacher models, using 
    different datasets for training. 

    This suppose a BERT based model
    """

    def __init__(self, 
                 pretrained_model_name='bert-base-uncased',
                 num_classes=3, 
                 use_pooler=False,
                 use_activation=False,
                 activation_function=torch.nn.ReLU,
                 use_dropout=False,
                 dropout_rate=0.1,
                 ):
        super(MultiHeadStudent, self).__init__()
        self.encoder = BertModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.use_activation = use_activation
        self.use_dropout = use_dropout
        self.use_pooler = use_pooler

        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        # We use one classification head to classes : non hate, implicit hate, explicit hate
        self.classification_head = torch.nn.Linear(self.encoder.config.hidden_size, num_classes)

        # Regression head gives us the rasch score
        self.regression_head = torch.nn.Linear(self.encoder.config.hidden_size, 1)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)


    def forward(self, input_ids, from_batch, attention_mask=None):
        model_out = self.encoder(input_ids, attention_mask=attention_mask)


        if self.use_pooler:
            embedding = model_out.pooler_output
        else:
            embedding = model_out.last_hidden_state[:, 0, :]
            if self.use_activation:
                embedding = self.activation_function(embedding)

        if self.use_dropout:
            dropout = torch.nn.Dropout(self.dropout_rate)
            embedding = dropout(embedding)

        if from_batch == 'classification':
            logits =  self.classification_head(embedding)
            return {'logits' : logits, 'embedding' : embedding}
        
        elif from_batch == 'regression':
            logits = self.regression_head(embedding)
            return {'logits' : logits, 'embedding' : embedding}
        else :
            raise ValueError("The task shoudl either be 'classification' or 'regression'!")