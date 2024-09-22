# coding: utf-8
import torch
import torch.nn as nn
from common.layers import TransformerEncoder, AttentionMechanism

class BD_TransformerLeNet(nn.Module):
    def __init__(self, text_config, lenet_model):
        super(BD_TransformerLeNet, self).__init__()

        # Use Transformer Encoder for text
        self.text_model = TransformerEncoder(text_config)

        # Use LeNet for image
        self.img_model = lenet_model

        # Attention Mechanism
        self.attention = AttentionMechanism(text_config.hidden_size + 768, 512)

        # Classifier layer
        self.classifier = nn.Linear(text_config.hidden_size + 768, 2)

        # Dropout
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, images):
        # Text model forward
        text_hidden_state = self.text_model(input_ids)

        # Image model forward
        img_hidden_state = self.img_model(images)

        # Concat text and image result
        concat_hidden_state = torch.cat((text_hidden_state[:, 0, :], img_hidden_state), dim=1)

        # Apply attention mechanism
        context_vector, attention_weights = self.attention(concat_hidden_state.unsqueeze(1))

        # Apply dropout
        context_vector = self.dropout(context_vector)

        # Pass through classifier
        logits = self.classifier(context_vector)

        return logits