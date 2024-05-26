import logging
import string
import random
import copy

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from vislinginstruct.common.registry import registry
from vislinginstruct.models.blip2 import Blip2Base, disabled_train
from vislinginstruct.models.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


@registry.register_model("my_mmlm_flant5")
class MyMMLMFlanT5Instruct(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5": "configs/models/my_mmlm_flant5.yaml",
        "eval_flant5": "../eval_configs/eval_flant5.yaml"
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        vit_model_path='',
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        q_former_model='',
        q_former_config='', 
        freeze_qformer=True,
        llm_model="google/flan-t5-xl",
        fc_layer='',
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        num_few_shot_examples=0,
        few_shot_prob=0,
        qformer_text_input=True,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer(q_former_config, truncation_side="left")

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, vit_model_path
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading t5')
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(llm_model, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(llm_model, truncation_side='right')

        t5_config = T5Config.from_pretrained(llm_model)
        t5_config.dense_act_fn = "gelu"
        self.llm_model = T5ForConditionalGeneration.from_pretrained(
            llm_model, config=t5_config
        )

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()
        print('Loading t5 Done')


        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            q_former_config, num_query_token, self.visual_encoder.num_features
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")

        if fc_layer:
            self.load_from_pretrained(url_or_filename=fc_layer)

        print('Loading Q-Former Done')

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt

        self._lemmatizer = None

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_prob = few_shot_prob

        self.qformer_text_input = qformer_text_input
        
    def forward(self, samples):

        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        # inputs_llm_proj = torch.concat((query_output.last_hidden_state[:,:query_tokens.size(1)+1,:], 
        #                                 torch.mean(query_output.last_hidden_state[:,query_tokens.size(1)+1:,:], dim=1, keepdim=True)), 
        #                                 dim=1)
        # inputs_llm = self.t5_proj(inputs_llm_proj)
        inputs_llm = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        fs_embeds, fs_atts = None, None
        if self.few_shot_prob > 0 and "few_shot_samples" in samples.keys():
            fs_embeds, fs_atts = self.prepare_few_shot_embeds(samples['few_shot_samples'])

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_output_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_llm, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.llm_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)

            if fs_embeds is not None:
                inputs_embeds = torch.cat([fs_embeds, inputs_embeds], dim=1)
                encoder_atts = torch.cat([fs_atts, encoder_atts], dim=1)

            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return {"loss": loss}


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]
        
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

        # For video data
        if image.dim() == 5:
            inputs_t5, atts_t5 = [], []
            add_inputs_llm, add_atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)
                    frame_features =self.visual_encoder.get_intermediate_layers(this_frame)[-2]
                    
                frame_features = frame_features[:, 1:] 
        
                add_feature_llm = self.vision_project(frame_features) 
                atts_add_feature_llm = torch.ones(add_feature_llm.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask = Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_t5 = self.t5_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_t5 = torch.ones(frame_inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
                inputs_t5.append(frame_inputs_t5)
                atts_t5.append(frame_atts_t5)
                add_inputs_llm.append(add_feature_llm)
                add_atts_llm.append(atts_add_feature_llm)
            inputs_t5 = torch.cat(inputs_t5, dim=1)
            atts_t5 = torch.cat(atts_t5, dim=1)
            add_feature_llm = torch.cat(add_inputs_llm, dim=1)
            atts_add_feature_llm = torch.cat(add_atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            
            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            # inputs_llm_proj = torch.concat((query_output.last_hidden_state[:,:query_tokens.size(1)+1,:], 
            #                                 torch.mean(query_output.last_hidden_state[:,query_tokens.size(1)+1:,:], dim=1, keepdim=True)), 
            #                                 dim=1)
            # inputs_llm = self.t5_proj(inputs_llm_proj)
            inputs_llm = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_llm, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.llm_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                    for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"][i],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - prompt: the instruction
            candidates:
                (list): A list of candidate class names;
            n_segments:
                (int): Split the candidates into n_segments and predict one by one. This is useful when the number of candidates is too large.
        Returns:
            output_class: predicted class index
        """

        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            if bs == 1:
                samples["context"] = [samples["context"]]
            prompt = [f'Context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            if bs == 1:
                samples["history"] = [samples["history"]]
            prompt = [f'Dialog history: {samples["history"][i]}\n {prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_t5, atts_t5 = [], []
            add_inputs_llm, add_atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)
                    frame_features =self.visual_encoder.get_intermediate_layers(this_frame)[-2]
                    
                frame_features = frame_features[:, 1:] 
        
                add_feature_llm = self.vision_project(frame_features) 
                atts_add_feature_llm = torch.ones(add_feature_llm.size()[:-1], dtype=torch.long).to(image.device)
                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_t5 = self.t5_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_t5 = torch.ones(frame_inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
                inputs_t5.append(frame_inputs_t5)
                atts_t5.append(frame_atts_t5)
                add_inputs_llm.append(add_feature_llm)
                add_atts_llm.append(atts_add_feature_llm)
            inputs_t5 = torch.cat(inputs_t5, dim=1)
            atts_t5 = torch.cat(atts_t5, dim=1)
            add_feature_llm = torch.cat(add_inputs_llm, dim=1)
            atts_add_feature_llm = torch.cat(add_atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            # inputs_llm_proj = torch.concat((query_output.last_hidden_state[:,:query_tokens.size(1)+1,:], 
            #                                 torch.mean(query_output.last_hidden_state[:,query_tokens.size(1)+1:,:], dim=1, keepdim=True)), 
            #                                 dim=1)
            # inputs_llm = self.t5_proj(inputs_llm_proj)
            inputs_llm = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)
        output_tokens = self.t5_tokenizer(
            candidates, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_llm, input_tokens.attention_mask], dim=1)

        n_cands = len(candidates)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.llm_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)

            encoder_outputs = self.llm_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
            )

            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                # this_encoder_outputs = copy.deepcopy(encoder_outputs)
                this_encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0].clone(),
                )

                this_encoder_outputs['last_hidden_state'] = this_encoder_outputs[0].repeat_interleave(seg_len, dim=0)
                this_encoder_atts = encoder_atts.repeat_interleave(seg_len, dim=0)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len
                this_output_tokens_ids = output_tokens.input_ids[start_i:end_i].repeat(bs, 1)
                this_output_tokens_atts = output_tokens.attention_mask[start_i:end_i].repeat(bs, 1)

                this_targets = this_output_tokens_ids.masked_fill(this_output_tokens_ids == self.t5_tokenizer.pad_token_id, -100)

                outputs = self.llm_model(
                    encoder_outputs=this_encoder_outputs,
                    attention_mask=this_encoder_atts,
                    decoder_attention_mask=this_output_tokens_atts,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )
                loss = outputs.loss.tolist()

                # loss = loss.reshape(bs, seg_len)
                all_losses += loss
                # output_class_ranks = torch.argsort(loss, dim=-1)
                # all_losses.append(loss)

            # all_losses = torch.cat(all_losses, dim=-1)
            # output_class_ranks = torch.argsort(all_losses, dim=-1)[0]
            output_class_ranks = [i[0] for i in sorted(enumerate(all_losses), key=lambda x:x[1])]

        return output_class_ranks

    def calculate_ias(self, samples):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."
        
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        # inputs_llm_proj = torch.concat((query_output.last_hidden_state[:,:query_tokens.size(1)+1,:], 
        #                                 torch.mean(query_output.last_hidden_state[:,query_tokens.size(1)+1:,:], dim=1, keepdim=True)), 
        #                                 dim=1)
        # inputs_llm = self.t5_proj(inputs_llm_proj)
        inputs_llm = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        fs_embeds, fs_atts = None, None
        if self.few_shot_prob > 0 and "few_shot_samples" in samples.keys():
            fs_embeds, fs_atts = self.prepare_few_shot_embeds(samples['few_shot_samples'])

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_output_tokenizer(
                samples["prompt"],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_llm, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.llm_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)

            if fs_embeds is not None:
                inputs_embeds = torch.cat([fs_embeds, inputs_embeds], dim=1)
                encoder_atts = torch.cat([fs_atts, encoder_atts], dim=1)

            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return loss

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        vit_model_path = cfg.get("vit_model_path")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")

        q_former_model = cfg.get("q_former_model")
        q_former_config = cfg.get("q_former_config")
        freeze_qformer = cfg.get("freeze_qformer")
        llm_model = cfg.get("llm_model")
        fc_layer = cfg.get("fc_layer")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        num_few_shot_examples = cfg.get("num_few_shot_examples", 0)
        few_shot_prob = cfg.get("few_shot_prob", 0.0)

        qformer_text_input = cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            vit_model_path=vit_model_path,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            q_former_model=q_former_model,
            q_former_config=q_former_config, 
            freeze_qformer=freeze_qformer,
            llm_model=llm_model,
            fc_layer=fc_layer, 
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_prob=few_shot_prob,
            qformer_text_input=qformer_text_input,
        )

        # model.load_checkpoint_from_config(cfg)

        return model
