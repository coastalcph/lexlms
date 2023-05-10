# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model."""

import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)

import copy
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Stack


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.Tanh()
        classifier_dropout = (
            config.dropout_rate if config.dropout_rate is not None else 0.0
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(self,
                hidden_states: Optional[torch.FloatTensor],
                attention_mask: Optional[torch.FloatTensor] = None,
                eos_position_indices: torch.Tensor = -1) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the last eos token.
        last_token_tensor = hidden_states[eos_position_indices]
        pooled_output = self.dense(last_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class LabelWiseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.d_model // config.num_attention_heads
        self.all_head_size = config.d_model
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)

        classifier_dropout = (
            config.dropout_rate if config.dropout_rate is not None else 0.0
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.label_encodings = nn.Parameter(torch.Tensor(self.num_labels, config.d_model),
                                            requires_grad=True)

        self.classifier = nn.Linear(config.d_model, 1, bias=False)

        # init label-related matrices
        self.label_encodings.data.normal_(mean=0.0, std=0.02)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states: Optional[torch.FloatTensor],
                attention_mask: Optional[torch.FloatTensor] = None,
                eos_position_indices: torch.Tensor = None,
                ) -> torch.Tensor:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(torch.unsqueeze(self.label_encodings, 0).repeat(hidden_states.size(0), 1, 1))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # attention_mask = self.get_extended_attention_mask(attention_mask)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # Compute label scores / outputs
        return self.classifier(context_layer).squeeze()


class T5ForSequenceClassificatiom(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        use_lwan = False
        self.model_dim = config.d_model
        self.num_labels = config.num_labels
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        if use_lwan:
            self.classifier = LabelWiseAttention(config)
        else:
            self.classifier = Pooler(config)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def tie_weights(self):
        return

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Convert encoder inputs in embeddings if needed
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        eos_position_indices = (input_ids == self.config.eos_token_id)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.size())

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output,
                                 attention_mask=extended_attention_mask,
                                 eos_position_indices=eos_position_indices)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else outputs

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer
    model = T5ForSequenceClassificatiom.from_pretrained('t5-small')
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    inputs = tokenizer(['dog ' * 500] * 3, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
    model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=torch.zeros(len(inputs['input_ids']), 2))
    print()
