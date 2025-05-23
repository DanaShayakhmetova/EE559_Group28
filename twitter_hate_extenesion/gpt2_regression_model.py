from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, GPT2PreTrainedModel, get_scheduler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from torch import nn
import torch

class GPT2Regression(GPT2PreTrainedModel):
    def __init__(self, config): 
        super().__init__(config)
        self.gpt2 = GPT2Model(config)
        self.reg_head = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        last_id = attention_mask.sum(dim=1) - 1
        
        pooled = last_hidden[torch.arange(last_hidden.size(0)), last_id]
        pred = self.reg_head(pooled).squeeze(-1)
        loss = None

        if labels is not None : 
            loss_fn = nn.MSELoss()
            loss = loss_fn(pred, labels)

        return {"loss" : loss, "logits" : pred}
    
    @staticmethod
    def load_model(weight_path):
        """
        Loads the model from the specified path.
        Args:
            weight_path (str): Path to the model weights.
        
        Returns:
            tokenizer (GPT2Tokenizer): The tokenizer used for the model.
            config (GPT2Config): The configuration of the model.
            model (GPT2Regression): The loaded model.
        """
        tokenizer = GPT2Tokenizer.from_pretrained(weight_path)
        config = GPT2Config.from_pretrained(weight_path)
        model = GPT2Regression.from_pretrained(weight_path)
        
        return tokenizer, config, model