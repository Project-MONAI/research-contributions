#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor as T
from transformers import BertForMaskedLM
from transformers.modeling_outputs import ModelOutput

from .configuration_cxrbert import CXRBertConfig

BERTTupleOutput = Tuple[T, T, T, T, T]

class CXRBertOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    logits: torch.FloatTensor
    cls_projected_embedding: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertProjectionHead(nn.Module):
    '''
    Projection head to be used with BERT CLS token, it's similar to `BertPredictionHeadTransform` in HuggingFace library.
    :param config: CXRBertConfig
    :return: (batch_size, output_size)
    '''
    def __init__(self, config: CXRBertConfig) -> None:
        super().__init__()
        self.dense_to_hidden = nn.Linear(config.hidden_size, config.projection_size)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(config.projection_size, eps=1e-12)
        self.dense_to_output = nn.Linear(config.projection_size, config.projection_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_to_hidden(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense_to_output(hidden_states)

        return hidden_states


class CXRBertModel(BertForMaskedLM):
    """
    Implements the CXR-BERT model outlined in the manuscript:
    Boecking et al. "Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing", 2022
    https://arxiv.org/abs/2204.09817

    Extends the HuggingFace BertForMaskedLM model by adding a separate projection head. The projection "[CLS]" token is used to align
    the latent vectors of image and text modalities.
    """

    config_class = CXRBertConfig

    def __init__(self, config: CXRBertConfig):
        super().__init__(config)

        self.cls_projection_head = BertProjectionHead(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.projection_size, 2)
        
        self.loss_fct = nn.CrossEntropyLoss()
        
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_cls_projected_embedding: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        labels = None,
        class_labels = None,
        **kwargs: Any
    ) -> Union[BERTTupleOutput, CXRBertOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #print(labels)
        bert_for_masked_lm_output = super().forward(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids,
                                                    position_ids=position_ids,
                                                    head_mask=head_mask,
                                                    inputs_embeds=inputs_embeds,
                                                    output_attentions=output_attentions,
                                                    labels=labels,
                                                    output_hidden_states=True,
                                                    return_dict=True)
        #print(type(bert_for_masked_lm_output))
        return bert_for_masked_lm_output
        last_hidden_state = bert_for_masked_lm_output.hidden_states[-1]
        cls_projected_embedding = self.cls_projection_head(last_hidden_state[:, 0, :]) if output_cls_projected_embedding else None
        seq_relationship_scores = self.classifier(cls_projected_embedding)
        lm_loss = bert_for_masked_lm_output.loss
        
        #print(class_labels)
        if class_labels is not None:    
            next_sentence_loss = self.loss_fct(seq_relationship_scores.view(-1, 2), class_labels.view(-1))
        else:
            next_sentence_loss = 0.

        if return_dict:
            #print(bert_for_masked_lm_output.logits)
            return CXRBertOutput(
                loss = lm_loss + next_sentence_loss,
                last_hidden_state=last_hidden_state,
                logits=bert_for_masked_lm_output.logits,
                cls_projected_embedding=cls_projected_embedding,
                hidden_states=bert_for_masked_lm_output.hidden_states if output_hidden_states else None,
                attentions=bert_for_masked_lm_output.attentions,
                labels=labels
            )
        else:
            return (
                lm_loss + next_sentence_loss,
                last_hidden_state,
                bert_for_masked_lm_output.logits,
                cls_projected_embedding,
                bert_for_masked_lm_output.hidden_states,
                bert_for_masked_lm_output.attentions,)

    def get_projected_text_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns l2-normalised projected cls token embeddings for the given input token ids and attention mask.
        The joint latent space is trained using a contrastive objective between image and text data modalities.

        :param input_ids: (batch_size, sequence_length)
        :param attention_mask: (batch_size, sequence_length)
        :return: (batch_size, projection_size)
        """

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, 
                               output_cls_projected_embedding=True, return_dict=True)
        assert isinstance(outputs, CXRBertOutput)

        normalized_cls_embedding = F.normalize(outputs.cls_projected_embedding, dim=1)
        return normalized_cls_embedding
