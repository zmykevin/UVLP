# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/uclanlp/visualbert
# which was cleaned up and adapted for MMF.

import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

#added by Mingyang
import numpy as np
import json

import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.embeddings import BertVisioLinguisticEmbeddings, BertVisioLinguisticImageTagEmbeddings
from mmf.modules.hf_layers import BertEncoderJit, BertLayerJit
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from mmf.utils.torchscript import getattr_torchscriptable
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_bert import (
    BertConfig,
    BertForPreTraining,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel
)

#########Added for visual masked language modeling####################
def load_output_header(output_header, bert):
  target = dict(output_header.named_parameters())
  source = dict(bert.named_parameters())
  parts = list(target.keys())


  for x in parts:
    target_key = x
    source_key = "cls."+target_key
    if source.get(source_key, None) is not None:
        target[target_key].data.copy_(source[source_key].data)  
class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val

class ItmHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    # def init_weights(self, module)
    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class BertVisualObjHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        self.visual_losses = config.visual_losses

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict(
            {
                key: nn.Linear(config.hidden_size, config.visual_loss_config[key][0])
                for key in self.visual_losses
            }
        )
        #self.decoder = nn.Linear(config.hidden_size, config.visual_embedding_dim)
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        #output = self.decoder(hidden_states)
        return output

# class BertVisualObjClassificationHead(nn.Module):
#         super().__init__()
#         self.transform = BertPredictionHeadTransform(config)
#         self.decoder = nn.Linear(config.hidden_size, config.visual_embedding_dim)
#     def forward(self, hidden_states):
#         hidden_states = self.transform(hidden_states)
#         # output = {}
#         # for key in self.visual_losses:
#         #     output[key] = self.decoder_dict[key](hidden_states)
#         output = self.decoder(hidden_states)
#         return output

########################################################################

class VisualBERTBase(BertPreTrainedModel):
    def __init__(
        self,
        config,
        visual_embedding_dim=512,
        embedding_strategy="plain",
        bypass_transformer=False,
        output_attentions=False,
        output_hidden_states=False,
        image_tag_pretraining = False,
    ):
        super().__init__(config)
        self.config = config

        config.visual_embedding_dim = visual_embedding_dim
        config.embedding_strategy = embedding_strategy
        config.bypass_transformer = bypass_transformer
        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states
        
        # if not image_tag_pretraining:
        #     self.embeddings = BertVisioLinguisticEmbeddings(config)
        # else:
        self.embeddings = BertVisioLinguisticImageTagEmbeddings(config)
        self.encoder = BertEncoderJit(config)
        self.pooler = BertPooler(config)
        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = BertLayerJit(config)

        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        image_text_alignment: Optional[Tensor] = None,
        visual_tag_box: Optional[Tensor] = None,
        box: Optional[Tensor] = None,


    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # Python builtin next is currently not supported in Torchscript
        if not torch.jit.is_scripting():
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids,
            token_type_ids,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type,
            image_text_alignment=image_text_alignment,
            visual_tag_box=visual_tag_box,
            box=box
        )

        if (
            self.bypass_transformer
            and visual_embeddings is not None
            and hasattr(self, "additional_layer")
        ):
            assert (
                not self.output_hidden_states
            )  # Don't support this for the bypass model
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_part = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[
                :, :, :text_length, :text_length
            ]

            encoded_layers = self.encoder(
                text_embedding_output, text_extended_attention_mask
            )
            sequence_output = encoded_layers[0]
            new_input = torch.cat((sequence_output, visual_part), dim=1)
            final_sequence_output = self.additional_layer(
                new_input, extended_attention_mask
            )
            pooled_output = self.pooler(final_sequence_output[0])
            return final_sequence_output[0], pooled_output, []

        else:
            encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_attentions=self.output_attentions)
            #Added by Mingyang
            sequence_output = encoded_layers[0]
            pooled_output = self.pooler(sequence_output)
            #if masked_lm_labels is not None:
            
            # else:
            #     pooled_output = None
            attn_data_list: List[Tensor] = []

            if not torch.jit.is_scripting():
                if self.output_attentions:
                    attn_data_list = encoded_layers[1:]
            else:
                assert (
                    not self.output_attentions
                ), "output_attentions not supported in script mode"

            #Save Num Boxes
            #Save Bbox
            return sequence_output, pooled_output, attn_data_list
            #return lang_feats, visn_feats


class VisualBERTForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.visual_losses = self.config.visual_losses
        self.visual_loss_config = self.config.visual_loss_config

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = self.config.get("bert_model_name", None)
        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )
        if self.bert_model_name is None:
            self.bert = VisualBERTBase(
                self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            self.bert = VisualBERTBase.from_pretrained(
                self.config.bert_model_name,
                config=self.bert_config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )

        self.vocab_size = self.bert.config.vocab_size

        # TODO: Once omegaconf fixes int keys issue, bring this back
        # See https://github.com/omry/omegaconf/issues/149
        # with omegaconf.open_dict(self.config):
        #     self.config.update(self.bert.config.to_dict())

        if self.bert_model_name is None:
            bert_masked_lm = BertForPreTraining(self.bert.config)
        else:
            bert_masked_lm = BertForPreTraining.from_pretrained(
                self.config.bert_model_name,
                config=self.bert.config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
            )
        
        self.cls = BertOnlyMLMHead(self.bert.config, self.bert.embeddings.word_embeddings.weight)
        #load the weights of cls
        load_output_header(self.cls, bert_masked_lm)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        if self.config.task_visn:
            #self.visn_loss = nn.SmoothL1Loss(reduction="none")
            self.visn_loss = {
            "l2": nn.SmoothL1Loss(reduction="none"),
            "ce": nn.CrossEntropyLoss(ignore_index=-1, reduction="none"),
            }
            self.obj_predict_head = BertVisualObjHead(self.config)

        if self.config.task_matched:
            self.itm_cls = deepcopy(bert_masked_lm.cls.seq_relationship)
        #     self.matched_loss = nn.CrossEntropyLoss(ignore_index=-1)
            self.softmax = nn.Softmax(dim=1)
            #self.fake_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

        if self.config.itm_filtering:
            assert self.config.task_matched, "task_matched has to be enabled for this setting"
            self.loss_fct_noreduce = nn.CrossEntropyLoss(ignore_index=-1)
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)
                self.itm_cls.apply(self.bert._init_weights)

            self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them
            instead.
        """
        self.bert._tie_or_clone_weights(
            self.cls.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def compute_itm_weights(self, seq_relationship_score, matched_label):
        prediction_score = self.softmax(seq_relationship_score)
        prediction_weight = prediction_score[:,1] * matched_label + (1-matched_label)
        return prediction_weight


    def forward(
        self,
        input_ids: Tensor,
        input_mask: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        image_text_alignment: Optional[Tensor] = None,
        masked_lm_labels: Optional[Tensor] = None,
        masked_image_labels: Optional[Tensor] = None,
        obj_labels: Optional[Tensor] = None,
        mrtm_labels: Optional[Tensor] = None,
        visual_tag_box: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        matched_label: Optional[Tensor] = None,
        num_boxes: Optional[list] = None,
        object_mask: Optional[Tensor] = None,
        disable_itm=True,
        disable_itm_filtering=True,
        current_epoch=1,
        loaded_batch=1,
    ) -> Dict[str, Tensor]:
        #Initialize the masked_img_loss as None
        masked_img_loss: Optional[Tensor] = None
        #################################################
        sequence_output, pooled_output, attention_weights = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            visual_embeddings_type,
            image_text_alignment,
            visual_tag_box,
            box,
        )
        
        #print(pooled_output)
        output_dict: Dict[str, Tensor] = {}
        if not torch.jit.is_scripting():
            if self.output_attentions:
                output_dict["attention_weights"] = attention_weights

            if self.output_hidden_states:
                output_dict["sequence_output"] = sequence_output
                output_dict["pooled_output"] = pooled_output
        else:
            assert not (
                self.output_attentions or self.output_hidden_states
            ), "output_attentions or output_hidden_states not supported in script mode"
        


        if masked_lm_labels is not None and disable_itm:
            prediction_scores = self.cls(sequence_output)
            output_dict["logits"] = prediction_scores

            if self.config.itm_filtering and (current_epoch >= self.config.itm_filtering_start_epoch) and not disable_itm_filtering:
                with torch.no_grad():
                    seq_relationship_score = self.itm_cls(pooled_output)
                    psudo_matched_label = torch.ones_like(seq_relationship_score[:,0])
                    #print(psudo_matched_label)
                    prediction_weight = self.compute_itm_weights(seq_relationship_score, psudo_matched_label)
                masked_lm_loss = self.loss_fct_noreduce(
                    prediction_scores.contiguous().view(-1, self.vocab_size),
                    masked_lm_labels.contiguous().view(-1),
                )
                masked_lm_loss = (masked_lm_loss*prediction_weight).mean()

            else:
                masked_lm_loss = self.loss_fct(
                    prediction_scores.contiguous().view(-1, self.vocab_size),
                    masked_lm_labels.contiguous().view(-1),
                )

            output_dict["masked_lm_loss"] = masked_lm_loss
            output_dict["loss"] = masked_lm_loss

        #Add a visual_prediction output scores for mrtm
        if masked_image_labels is not None and mrtm_labels is not None and self.config.task_mrtm:
            visn_output = sequence_output[:,-masked_image_labels.size(1):]
            prediction_scores = self.cls(visn_output)
            mrtm_loss = self.loss_fct(prediction_scores.contiguous().view(-1, self.vocab_size),mrtm_labels.contiguous().view(-1))
            output_dict["mrtm_loss"] = mrtm_loss

        #Add a visual_prediction_output scores
        if masked_image_labels is not None and self.config.task_visn:
            #implement mrfr, moc loss
            total_visn_loss = 0.0
            visn_output = sequence_output[:,-masked_image_labels.size(1):]
            #pooled_output.detach() #detach pooled output
            visn_prediction_scores_dict = self.obj_predict_head(visn_output)
            for key in self.visual_losses:
                visn_prediction_scores = visn_prediction_scores_dict[key]
                (
                  output_dim,
                  loss_fct_name,
                  label_shape,
                  weight,
                ) = self.visual_loss_config[key]
                mask_conf = (masked_image_labels == 1).float() #the mask_conf for object can be changed later
                if key == "feat":
                    if type(masked_image_labels) is None:
                        continue
                    visn_loss = self.visn_loss[loss_fct_name](
                        visn_prediction_scores.view(-1, output_dim), 
                        visual_embeddings.view(-1, output_dim)
                        )

                if key == "obj":
                    if (type(masked_image_labels) is None) or (mrtm_labels is not None):
                        continue
                    visn_loss = self.visn_loss[loss_fct_name](
                        visn_prediction_scores.view(-1, output_dim),
                        obj_labels.view(-1)
                        )
                
                if visn_loss.dim() > 1:
                    visn_loss = visn_loss.mean(1)
                visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
                total_visn_loss += visn_loss
            output_dict["visn_loss"] = total_visn_loss

        #Add a image text matching
        if matched_label is not None and self.config.task_matched and not disable_itm:
            seq_relationship_score = self.itm_cls(pooled_output)
            matched_label = matched_label.to(seq_relationship_score).long()
            
            #Implemented by Mingyang
            if self.config.itm_filtering and (current_epoch >= self.config.itm_filtering_start_epoch):
                # print("itm filtering starts")
                with torch.no_grad():
                    prediction_weight = self.compute_itm_weights(seq_relationship_score, matched_label)
                matched_loss = self.loss_fct_noreduce(seq_relationship_score.view(-1, 2), matched_label)
                matched_loss = prediction_weight * matched_loss
                matched_loss = matched_loss.mean()
            else:
                matched_loss = self.loss_fct(seq_relationship_score.view(-1, 2), matched_label)
            # with torch.no_grad():

            output_dict["matched_loss"] = matched_loss
            output_dict["scores"] = seq_relationship_score
        else:
            #give back a psudo scores
            output_dict["scores"] = torch.zeros_like(matched_label)
        # print(output_dict)
        return output_dict

class VisualBERTForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.pooler_strategy = self.config.get("pooler_strategy", "default")

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = getattr(self.config, "bert_model_name", None)
        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )
        if self.bert_model_name is None:
            self.bert = VisualBERTBase(
                self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            self.bert = VisualBERTBase.from_pretrained(
                self.config.bert_model_name,
                config=self.bert_config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )

        self.training_head_type = self.config.training_head_type
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        #Define special output for refcoco
        if self.config.training_head_type == "refcoco":
            #Define by Mingyang, following Uniter's setting
            self.re_output = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.GELU(),
                nn.LayerNorm(self.config.hidden_size, eps=1e-12),
                nn.Linear(self.config.hidden_size, 1)
                )
            #self.re_output = nn.Linear(self.config.hidden_size,1)
        elif self.config.training_head_type == "itm_flickr30k":
            if self.bert_model_name is None:
                bert_masked_lm = BertForPreTraining(self.bert.config)
            else:
                bert_masked_lm = BertForPreTraining.from_pretrained(
                    self.config.bert_model_name,
                    config=self.bert.config,
                    cache_dir=os.path.join(
                        get_mmf_cache_dir(), "distributed_{}".format(-1)
                    ),
                )
            #self.itm_cls = nn.Linear(self.bert.config.hidden_size, 2) #can be initialized from pre-trained model
            # self.rank_cls = nn.Linear(self.bert.config.hidden_size,1)
            #initialize the margin value
            #self.margin = 0.2 #Hardcoded here. 
            self.itm_cls = deepcopy(bert_masked_lm.cls.seq_relationship)
        else:
            if self.config.training_head_type == "nlvr2":
                self.bert.config.hidden_size *= 2

            self.classifier = nn.Sequential(
                BertPredictionHeadTransform(self.bert.config),
                nn.Linear(self.bert.config.hidden_size, self.config.num_labels),
            )
        
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()
                self.itm_cls.apply(self.bert._init_weights) #initialize itm_cls

            # Classifier needs to be initialized always as it is task specific
            if self.config.training_head_type not in ["refcoco", "itm_flickr30k"]:
                self.classifier.apply(self.bert._init_weights)

        # Set last hidden layer
        if "losses" in self.config and self.config.zerobias:
            for loss in self.config.losses:
                if "bce" in loss["type"]:
                    self.classifier[1].bias.data.fill_(self.config.biasfill)
    
    # def init_output(self):
    #     """ need to be called after from pretrained """
    #     self.rank_cls.weight.data = self.itm_cls.weight.data[1:, :]
    #     self.rank_cls.bias.data = self.itm_cls.bias.data[1:]

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        image_text_alignment: Optional[Tensor] = None,
        masked_lm_labels: Optional[Tensor] = None,
        masked_image_labels: Optional[Tensor] = None,
        obj_labels: Optional[Tensor] = None,
        mrtm_labels: Optional[Tensor] = None,
        visual_tag_box: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        matched_label: Optional[Tensor] = None,
        num_boxes: Optional[list] = None,
        object_mask: Optional[Tensor] = None,
        disable_itm=True,
        disable_itm_filtering=True,
        current_epoch=1,
        loaded_batch=1,
    ) -> Dict[str, Tensor]:
        sequence_output, pooled_output, attention_weights = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            visual_embeddings_type,
            image_text_alignment,
            visual_tag_box,
            box,
        )

        #input_mask is the input mask for input_ids
        if self.training_head_type == "refcoco":
            output_dict: Dict[str, Tensor] = {}
            sequence_output = self._get_image_hidden(sequence_output, num_boxes)
            #print(sequence_output)
            scores = self.re_output(sequence_output).squeeze(2)
            scores = scores.masked_fill(object_mask, -1e4) #mask out non-objects
            # print(scores.size())
            #return the scores
            output_dict["scores"] = scores
        elif self.training_head_type == "itm_flickr30k":
            output_dict: Dict[str, Tensor] = {} 
            
            ################Binary Classification Loss########################
            logits = self.itm_cls(pooled_output)
            #print(logits)
            reshaped_logits = logits.contiguous().view(-1, self.num_labels)
            output_dict["scores"] = reshaped_logits
        else:
            if self.training_head_type == "nlvr2":
                # 2B * H => B * 2H
                b, h = pooled_output.size()
                pooled_output = torch.cat(
                    [pooled_output[: b // 2], pooled_output[b // 2 :]], dim=1
                )

            output_dict: Dict[str, Tensor] = {}
            if not torch.jit.is_scripting():
                if self.output_attentions:
                    output_dict["attention_weights"] = attention_weights

                if self.output_hidden_states:
                    output_dict["sequence_output"] = sequence_output
                    output_dict["pooled_output"] = pooled_output
            else:
                assert not (
                    self.output_attentions or self.output_hidden_states
                ), "output_attentions or output_hidden_states not supported in script mode"

            if self.pooler_strategy == "vqa":
                # In VQA2 pooling strategy, we use representation from second last token
                index_to_gather = input_mask.sum(1) - 2
                pooled_output = torch.gather(
                    sequence_output,
                    1,
                    index_to_gather.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(index_to_gather.size(0), 1, sequence_output.size(-1)),
                )

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            reshaped_logits = logits.contiguous().view(-1, self.num_labels)
            # print(reshaped_logits.size())
            output_dict["scores"] = reshaped_logits
        return output_dict
    def _get_image_hidden(self, sequence_output, num_bbs, txt_len=60, max_bb=100):
        """
        Extracting the img_hidden part from sequence_output.
        Inputs:
        - sequence_output: (n, txt_len+num_bb, hid_size)
        - txt_lens       : [txt_len]
        - num_bbs        : [num_bb]
        Output:
        - img_hidden     : (n, max_num_bb, hid_size)
        """
        outputs = []
        hid_size = sequence_output.size(-1)
        for seq_out, nbb in zip(sequence_output.split(1, dim=0),num_bbs):
            img_hid = seq_out[:, txt_len:txt_len+nbb, :]
            if nbb < max_bb:
                img_hid = torch.cat(
                        [img_hid, self._get_pad(
                            img_hid, max_bb-nbb, hid_size)],
                        dim=1)
            outputs.append(img_hid)

        img_hidden = torch.cat(outputs, dim=0)
        return img_hidden

    def _get_pad(self, t, len_, hidden_size):
        pad = torch.zeros(1, len_, hidden_size, dtype=t.dtype, device=t.device)
        return pad


@registry.register_model("visual_bert")
class VisualBERT(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.training_head_type: str = self.config.training_head_type

    @classmethod
    def config_path(cls):
        return "configs/models/visual_bert/pretrain.yaml"

    def build(self):
        if self.training_head_type == "pretraining":
            self.model = VisualBERTForPretraining(self.config)
        else:
            self.model = VisualBERTForClassification(self.config)

        if self.config.special_visual_initialize:
            self.model.bert.embeddings.initialize_visual_from_pretrained()

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def flatten(
        self,
        sample_list: Dict[str, Tensor],
        to_be_flattened: List[str],
        to_be_flattened_dim: List[str],
    ) -> Dict[str, Tensor]:
        for key in to_be_flattened:
            # Make sure these keys are present or otherwise set these keys to None
            sample_list[key] = transform_to_batch_sequence(sample_list[key])
        for key in to_be_flattened_dim:
            sample_list[key] = transform_to_batch_sequence_dim(sample_list[key])
        return sample_list

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        sample_list["visual_embeddings_type"] = torch.zeros_like(
            sample_list["image_mask"]
        )
        attention_mask = torch.cat(
            (sample_list["input_mask"], sample_list["image_mask"]), dim=-1
        )
        sample_list["attention_mask"] = attention_mask

        if self.training_head_type == "pretraining":
            #Added a condition for masked_lm_labels
            if sample_list.get("masked_lm_labels", None) is not None:
                assert sample_list["masked_lm_labels"].size(-1) == sample_list[
                    "input_mask"
                ].size(-1)
                new_lm_labels = torch.ones_like(attention_mask) * -1
                size_masked_lm_labels = sample_list["masked_lm_labels"].size()
                assert len(size_masked_lm_labels) == 2
                new_lm_labels[
                    : size_masked_lm_labels[0], : size_masked_lm_labels[1]
                ] = sample_list["masked_lm_labels"]
                sample_list["masked_lm_labels"] = new_lm_labels
                
        return sample_list

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids", "token_type_ids", "input_mask", "image_mask"]
        to_be_flattened_dim = ["visual_embeddings"]

        if self.training_head_type == "pretraining":
            if sample_list.get("masked_lm_labels", None) is not None:
                to_be_flattened.append("masked_lm_labels")
            if sample_list.get("visual_tag_boxes", None) is not None:
                to_be_flattened_dim.append("visual_tag_boxes")
        #bbox should be included
        if sample_list.get("bbox", None) is not None:
            to_be_flattened_dim.append("bbox")
        
        # We want to convert everything into: batch x sequence_length x (dim).
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def update_sample_list_based_on_head(
        self, sample_list: Dict[str, Tensor], loaded_batch=1
    ) -> Dict[str, Tensor]:
        # print(sample_list["image_ids"])
        bert_input_ids = sample_list["input_ids"]
        bert_input_mask = sample_list["input_mask"]
        bert_input_type_ids = sample_list["segment_ids"]
        #Add the input_image_labels, Mingyang Zhou
        bert_image_labels = sample_list.get("image_labels", None)
        #print(bert_image_labels)
        bert_visual_tag_box = sample_list.get("visual_tag_boxes", None)
        #Add the object_labels, Mingyang
        bert_object_labels = sample_list.get("objects_ids", None)
        bert_mrtm_labels = sample_list.get("mrtm_labels", None)

        #define image numboxes
        image_numboxes = None
        if self.training_head_type == "nlvr2":
            assert bert_input_mask is not None
            if not torch.jit.is_scripting():
                bert_input_ids = torch.cat([bert_input_ids, bert_input_ids])
                bert_input_mask = torch.cat([bert_input_mask, bert_input_mask])
                bert_input_type_ids = torch.cat(
                    [bert_input_type_ids, bert_input_type_ids]
                )

                # image input
                img0 = getattr(sample_list, "img0", {})
                image_feat_variable_0 = getattr(img0, "image_feature_0", None)
                img1 = getattr(sample_list, "img1", {})
                image_feat_variable_1 = getattr(img1, "image_feature_0", None)
                image_feat_variable = torch.cat(
                    [image_feat_variable_0, image_feat_variable_1]
                )
                # print(image_feat_variable_0.shape)

                image_info = getattr(img0, "image_info_0", {})
                image_dim_variable_0 = getattr(image_info, "max_features", None)
                #add the image_bbox information
                image_bbox_0 = getattr(image_info, "bbox", None)

                image_info = getattr(img1, "image_info_0", {})
                image_dim_variable_1 = getattr(image_info, "max_features", None)
                image_bbox_1 = getattr(image_info, "bbox", None)
                image_dim_variable = torch.cat(
                    [image_dim_variable_0, image_dim_variable_1]
                )
                
                if image_bbox_0 is not None and image_bbox_1 is not None:
                    #image_bbox = image_bbox_0 + image_bbox_1
                    image_bbox = torch.cat(
                    [image_bbox_0, image_bbox_1]
                )
                else:
                    image_bbox = None
            else:
                raise RuntimeError("nlvr2 head doesn't support scripting as of now")

        else:

            if not torch.jit.is_scripting():
                image_info = getattr(sample_list, "image_info_0", {})
                image_bbox = getattr(image_info, "bbox", None)
                #get the number of bbox
                image_numboxes = getattr(image_info, "num_boxes", None)
                image_dim_variable = getattr(image_info, "max_features", None)
                image_feat_variable = getattr(sample_list, "image_feature_0", None)
                #Add box and visual_tag_box
            else:
                image_feat_variable = sample_list["image_feature_0"]
                image_info = getattr(sample_list, "image_info_0", {})
                #get the number of bbox
                image_numboxes = getattr(image_info, "num_boxes", None)
                image_bbox = getattr(image_info, "bbox", None)
                image_dim_variable = None
                image_bbox = None
                
        if image_dim_variable is None:
            image_dim_variable = sample_list["image_feature_0"].new_full(
                size=(image_feat_variable.size(0), 1),
                fill_value=image_feat_variable.size(1),
            )
        # print(image_bbox)
        sample_list["visual_embeddings"] = image_feat_variable
        sample_list["image_dim"] = image_dim_variable
        sample_list["input_ids"] = bert_input_ids
        sample_list["input_mask"] = bert_input_mask
        sample_list["token_type_ids"] = bert_input_type_ids
        sample_list["image_labels"] = bert_image_labels
        sample_list["visual_tag_box"] = bert_visual_tag_box
        sample_list["bbox"] = image_bbox
        max_features = torch.tensor(
            image_feat_variable.shape[1], dtype=torch.int
        )

        image_label_variable = getattr(sample_list, "image_labels", None)
        if  image_label_variable is not None:
            image_label_variable = image_label_variable[:, : max_features.item(), None]
            # image_label_variable = image_label_variable.unsqueeze(-1).to(device)
            image_label_variable = image_label_variable.unsqueeze(-1)

        sample_list["image_labels"] = image_label_variable

        #Add the matched label
        is_correct = getattr(sample_list, "is_correct", None)
        if is_correct is not None:
            if isinstance(is_correct, torch.Tensor):
                is_correct = is_correct
            else:
                is_correct = torch.tensor(is_correct)
        sample_list["matched_label"] = is_correct

        #add numboxes
        if image_numboxes is not None:
            sample_list["num_boxes"] = image_numboxes

        #add object labels:
        #print(bert_object_labels.shape)
        if bert_object_labels is not None:
            #bert_object_labels = torch.tensor(bert_object_labels)[:,: max_features.item(), None]
            bert_object_labels = bert_object_labels[:,: max_features.item(), None]
            sample_list["object_labels"] = bert_object_labels
        #print(bert_mrtm_labels)
        if bert_mrtm_labels is not None:
            bert_mrtm_labels = bert_mrtm_labels[:,: max_features.item(), None]
            sample_list["mrtm_labels"] = bert_mrtm_labels

        return sample_list

    def add_custom_params(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings = sample_list["visual_embeddings"]
        image_dim = sample_list["image_dim"]

        if self.training_head_type == "pretraining":
            # pretraining labels
            if sample_list["lm_label_ids"][0] is not None:
                sample_list["masked_lm_labels"] = sample_list["lm_label_ids"]
            if (self.config.task_visn or self.config.task_mrtm) and sample_list.get("image_labels", None) is not None:
                sample_list["masked_image_labels"] = sample_list["image_labels"]

        if sample_list.get("num_boxes", None) is not None:
            object_masks = []
            for num_box in sample_list["num_boxes"]:
                object_masks.append(torch.tensor([0]*num_box+[1]*(visual_embeddings.size(1)-num_box), dtype=torch.bool, device=visual_embeddings.device))
            object_masks = torch.stack(object_masks, dim=0)

            sample_list["object_mask"] = object_masks
        # Prepare Mask
        image_mask = torch.arange(
            visual_embeddings.size(-2), device=visual_embeddings.device
        ).expand(visual_embeddings.size()[:-1])

        if len(image_dim.size()) < len(image_mask.size()):
            if len(image_mask.size()) == 2:
                image_dim = image_dim.unsqueeze(-1)
            elif len(image_mask.size()) == 3:
                image_dim = image_dim.unsqueeze(-1).expand(-1,3).unsqueeze(-1)
            assert len(image_dim.size()) == len(image_mask.size())
        image_mask = image_mask < image_dim
        # print(image_mask)
        sample_list["image_mask"] = image_mask.long()

        return sample_list

    # Backward compatibility for code from original VisualBERT
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("bert.bert", "model.bert")
            .replace("bert.cls", "model.cls")
            .replace("bert.classifier", "model.classifier")
        )

    def forward(self, sample_list: Dict[str, Tensor], current_epoch=1, loaded_batch=0) -> Dict[str, Tensor]:
        if torch.jit.is_scripting():
            assert (
                "image_feature_0" in sample_list
            ), "Key 'image_feature_0' is required in TorchScript model"

        sample_list = self.update_sample_list_based_on_head(sample_list,loaded_batch=loaded_batch)
        sample_list = self.add_custom_params(sample_list)
        sample_list = self.flatten_for_bert(sample_list)
        sample_list = self.add_post_flatten_params(sample_list)
        
        if "itm" not in sample_list["dataset_name"] and sample_list["dataset_name"] != "masked_conceptual_captions_image":
            disable_itm = True 
        else:
            disable_itm = False

        #Get the dataset_name
        if sample_list["dataset_name"] in ["masked_conceptual_captions_image", "masked_conceptual_captions_text"]:
            disable_itm_filtering = True
        else:
            disable_itm_filtering = False
        
        

        output_dict = self.model(
            sample_list["input_ids"],
            sample_list["input_mask"],
            sample_list["attention_mask"],
            sample_list["token_type_ids"],
            sample_list["visual_embeddings"],
            sample_list["visual_embeddings_type"],
            getattr_torchscriptable(sample_list, "image_text_alignment", None),
            getattr_torchscriptable(sample_list, "masked_lm_labels", None),
            getattr_torchscriptable(sample_list, "masked_image_labels", None),
            getattr_torchscriptable(sample_list, "object_labels", None),
            getattr_torchscriptable(sample_list, "mrtm_labels", None),
            getattr_torchscriptable(sample_list, "visual_tag_box", None),
            getattr_torchscriptable(sample_list, "bbox", None),
            getattr_torchscriptable(sample_list, "matched_label", None),
            getattr_torchscriptable(sample_list, "num_boxes", None),
            getattr_torchscriptable(sample_list, "object_mask", None),
            disable_itm,
            disable_itm_filtering,
            current_epoch,
            loaded_batch
        )
       
        if self.training_head_type == "pretraining":
            if not torch.jit.is_scripting():
                loss_key = "{}/{}".format(
                    sample_list["dataset_name"], sample_list["dataset_type"]
                )
                output_dict["losses"] = {}
                if output_dict.get("masked_lm_loss", None) is not None:
                    #print(output_dict["masked_lm_loss"])
                    output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop(
                        "masked_lm_loss"
                        )
                if output_dict.get("visn_loss", None) is not None:
                    output_dict["losses"][loss_key + "/visn_loss"] = output_dict.pop("visn_loss")
                if output_dict.get("matched_loss", None) is not None:
                    # print(output_dict["matched_loss"])
                    output_dict["losses"][loss_key + "/matched_loss"] = output_dict.pop("matched_loss")
                if output_dict.get("mrtm_loss", None) is not None:
                    # print(output_dict["matched_loss"])
                    output_dict["losses"][loss_key + "/mrtm_loss"] = output_dict.pop("mrtm_loss")

            else:
                raise RuntimeError("Pretraining head can't be used in script mode.")
        return output_dict