# coding: utf-8
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

# RoBERTa Model
class BD_Roberta(nn.Module):
    def __init__(self, model_checkpoint, lora_r=8, lora_alpha=1, lora_dropout=0.1):
        super(BD_Roberta, self).__init__()

        # Pre-trained Model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

        # LoRA Config
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=["query", 'key', "value"],  # Apply LoRA to the attention layers
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # Apply LoRA to Pre-trained Layers
        self.model = get_peft_model(self.model, lora_config)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        outputs = self.dropout(outputs)
        return outputs