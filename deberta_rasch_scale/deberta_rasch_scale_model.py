from transformers import DebertaTokenizer, DebertaModel, DebertaConfig, PreTrainedModel
from torch import nn
import torch
import os

class DebertaRegression(PreTrainedModel):
    """
    DeBERTa version of the Rasch scale model for regression tasks.
    """
    config_class = DebertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaModel(config)
        self.reg_head = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0] 
        pred = self.reg_head(pooled).squeeze(-1)
        loss = None

        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(pred, labels)

        return {"loss": loss, "logits": pred}


    @staticmethod
    def load_model(weight_path, local_files_only=True):
        tokenizer = DebertaTokenizer.from_pretrained(weight_path, local_files_only=local_files_only)
        config = DebertaConfig.from_pretrained(weight_path, local_files_only=local_files_only)
        
        model = DebertaRegression(config)
        model.deberta = DebertaModel.from_pretrained(weight_path, config=config, local_files_only=local_files_only)

        reg_head = nn.Linear(config.hidden_size, 1)
        reg_head.load_state_dict(torch.load(os.path.join(weight_path, 'regression_head.pth')))
        model.reg_head = reg_head

        return tokenizer, config, model

    def save_weights(self, save_path):
        self.save_pretrained(save_path)
        self.deberta.save_pretrained(save_path)

        os.makedirs(save_path, exist_ok=True)
        torch.save(self.reg_head.state_dict(), os.path.join(save_path, 'regression_head.pth'))
