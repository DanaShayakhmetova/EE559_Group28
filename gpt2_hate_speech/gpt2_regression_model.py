from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, GPT2PreTrainedModel, get_scheduler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch import nn
import torch
import os

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
    def load_model(weight_path, local_files_only=True):
        """
        Loads the model from the specified path.
        Args:
            weight_path (str): Path to the model weights.
        
        Returns:
            tokenizer (GPT2Tokenizer): The tokenizer used for the model.
            config (GPT2Config): The configuration of the model.
            model (GPT2Regression): The loaded model.

        Notes:
            - The model, tokenizer and config must be in a local directory.
        """

        tokenizer = GPT2Tokenizer.from_pretrained(weight_path, local_files_only=local_files_only, )
        config = GPT2Config.from_pretrained(weight_path, local_files_only=local_files_only)
        
        model = GPT2Regression(config)
        model.gpt2 = GPT2Model.from_pretrained(weight_path, config=config, local_files_only=local_files_only)

        reg_head = nn.Linear(config.hidden_size, 1)
        reg_head.load_state_dict(torch.load(weight_path + '/regression_head.pth'))

        model.reg_head = reg_head

        return tokenizer, config, model
    
    def save_weights(self, save_path):
        """
        Saves the model to the specified path.
        Args:
            save_path (str): Path to save the model weights.
        """

        self.save_pretrained(save_path)
        self.gpt2.save_pretrained(save_path)

        reg_path = save_path + '/regression_head'
        if not os.path.exists(reg_path):
            os.makedirs(reg_path)
        
        torch.save(self.reg_head.state_dict(), save_path + '/regression_head.pth')


