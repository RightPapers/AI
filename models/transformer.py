# coding: utf-8
import torch.nn as nn
from common.layers import TransformerEncoder, AttentionMechanism
    
# Transformer Encoder Model
class BD_Transformer(nn.Module):
    def __init__(self, text_config):
        super(BD_Transformer, self).__init__()
        self.text_model = TransformerEncoder(text_config)
        self.attention = AttentionMechanism(text_config.hidden_size, 512)
        self.additional_layer = nn.Sequential(
            nn.Linear(text_config.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(512, 2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask):
        text_hidden_state = self.text_model(input_ids)  # [batch_size, seq_length, hidden_size]
        context_vector, attention_weights = self.attention(text_hidden_state)  # [batch_size, 512]
        context_vector = self.dropout(context_vector)  # [batch_size, 512]
        intermediate_output = self.additional_layer(context_vector)  # [batch_size, 512]
        logits = self.classifier(intermediate_output)  # [batch_size, 2]
        return logits