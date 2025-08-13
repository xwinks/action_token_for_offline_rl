# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class for evaluating Moto-GPT on Robot Manipulation Benchmarks."""
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image

class MotoGPT_PolicyWraper:
    def __init__(self,
                 policy,
                 variant,
                 latent_motion_decoding_kwargs,
                 lang_tokenizer
    ):
        """Constructor."""
        self.test_chunk_size = variant['test_chunk_size']
        self.is_gripper_binary = variant['is_gripper_binary']
        self.pred_discrete_arm_action = variant['pred_discrete_arm_action']
        self.lang_tokenizer = lang_tokenizer

        # Preprocess
        input_size = variant['rgb_shape']
        rgb_mean = variant['rgb_mean']
        rgb_std = variant['rgb_std']
        self.transform = T.Compose([
            T.Resize(input_size, interpolation=Image.BICUBIC),
            T.Normalize(rgb_mean, rgb_std)])

        self.policy = policy

        self.act_dim = variant['act_dim']
        self.seq_len = variant['seq_len']
        self.chunk_size = variant['chunk_size']
        self.mask_latent_motion_probability = variant['mask_latent_motion_probability']
        self.latent_motion_pred = variant['latent_motion_pred']
        self.per_latent_motion_len = variant['per_latent_motion_len']

        assert self.mask_latent_motion_probability == 0.0 or self.mask_latent_motion_probability == 1.0
        self.test_seq_len = math.ceil(self.test_chunk_size / self.chunk_size)
        self.latent_motion_decoding_kwargs = latent_motion_decoding_kwargs

        self.use_temporal_ensemble = variant['use_temporal_ensemble']
        
    @property
    def device(self):
        return self.policy.device

    def rgb_process(self, rgb):
        rgb = Image.fromarray(rgb)
        rgb = T.ToTensor()(rgb.convert('RGB'))
        rgb = self.transform(rgb)
        return rgb
        
    def reset(self):
        """Reset function."""
        self.rollout_step_counter = 0

        if self.use_temporal_ensemble:
            if self.pred_discrete_arm_action:
                self.action_buffer = np.zeros((self.test_chunk_size, self.test_chunk_size, (self.act_dim-1)*3+1))
            else:
                self.action_buffer = np.zeros((self.test_chunk_size, self.test_chunk_size, self.act_dim))
            self.action_buffer_mask = np.zeros((self.test_chunk_size, self.test_chunk_size), dtype=bool)

    def step(self, obs, goal):
        """Step function."""
        # Language
        lang_inputs = self.lang_tokenizer(goal, return_tensors='pt', padding=True)
        tokenized_text = lang_inputs.input_ids
        lang_attention_mask = lang_inputs.attention_mask

        # RGB
        rgb = self.rgb_process(obs['rgb_obs']['rgb_static'])
        rgb_data = rgb.unsqueeze(0).unsqueeze(0)

        # Latent action tokens
        if self.latent_motion_pred:
            latent_motion_ids_data = torch.zeros((1, self.seq_len, self.per_latent_motion_len)).long() # (1, t, per_latent_motion_len)
        else:
            latent_motion_ids_data = None

        # Attention mask
        attention_mask = torch.ones(1, self.seq_len).long()

        # Latent attention mask:
        if self.latent_motion_pred:
            latent_mask = attention_mask
        else:
            latent_mask = None

        # Forward pass
        tokenized_text = tokenized_text.to(self.device)
        lang_attention_mask = lang_attention_mask.to(self.device) if lang_attention_mask is not None else None
        rgb_data = rgb_data.to(self.device)
        attention_mask = attention_mask.to(self.device)
        latent_motion_ids_data = latent_motion_ids_data.to(self.device) if self.latent_motion_pred else None
        latent_mask = latent_mask.to(self.device) if self.latent_motion_pred else None

        latent_motion_decoding_kwargs = self.latent_motion_decoding_kwargs
        latent_step = 1
        latent_motion_decoding_kwargs['buffer_len'] = latent_step
        with torch.no_grad():
            prediction = self.policy(
                rgb=rgb_data, 
                language=tokenized_text,
                attention_mask=attention_mask,
                latent_motion_ids=latent_motion_ids_data,
                latent_mask=latent_mask,
                train=False,
                lang_attention_mask=lang_attention_mask,
                in_simulation=True,
                **latent_motion_decoding_kwargs,
        )

        if self.mask_latent_motion_probability == 0.0:
            latent_motion_ids_data[:, latent_step-1] = prediction['latent_motion_id_preds'].to(self.device)
            

            while latent_step < self.test_seq_len:
                latent_step += 1
                latent_motion_decoding_kwargs['buffer_len'] = latent_step
                with torch.no_grad():
                    prediction = self.policy(
                        rgb=rgb_data, 
                        language=tokenized_text,
                        attention_mask=attention_mask,
                        latent_motion_ids=latent_motion_ids_data,
                        latent_mask=latent_mask,
                        train=False,
                        lang_attention_mask=lang_attention_mask,
                        in_simulation=True,
                        **latent_motion_decoding_kwargs,
                )

        # Arm action
        arm_action_preds = prediction['arm_action_preds']  # (1, t, chunk_size, act_dim - 1)
        if self.pred_discrete_arm_action:
            arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1, 3)
        else:
            arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1)  # (t*chunk_size, act_dim - 1)

        # Gripper action
        gripper_action_preds = prediction['gripper_action_preds']  # (1, t, chunk_size, 1)
        gripper_action_preds = gripper_action_preds.view(-1, 1)  # (t*chunk_size, 1)

        
        # Use the first test_chunk_size action
        arm_action_pred = arm_action_preds[:self.test_chunk_size]  # (test_chunk_size, act_dim - 1)
        gripper_action_pred = gripper_action_preds[:self.test_chunk_size]  # (test_chunk_size, 1)

        if not self.use_temporal_ensemble:
            if self.is_gripper_binary:
                gripper_action_pred = ((gripper_action_pred > 0).float()) * 2.0 - 1.0
            
            if self.pred_discrete_arm_action:
                arm_action_pred = arm_action_pred.softmax(dim=-1).argmax(dim=-1)
                
            action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=-1)  # (test_chunk_size, act_dim)
            action_pred = action_pred.detach().cpu()

        else:
            # Shift action buffer
            self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
            self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
            self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
            self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
            self.action_buffer_mask = self.action_buffer_mask * np.fliplr(np.triu(np.ones(self.test_chunk_size))).astype(bool)


            # Add to action buffer
            if self.pred_discrete_arm_action:
                action = torch.cat((arm_action_pred.reshape(arm_action_pred.shape[0], -1), gripper_action_pred), dim=-1) # (t*chunk_size, (act_dim-1)*3+1)
            else:
                action = torch.cat((arm_action_pred, gripper_action_pred), dim=-1) # (t*chunk_size, act_dim)
            action = action.detach().cpu().numpy()
            self.action_buffer[0] = action
            self.action_buffer_mask[0] = True
            
            # Ensemble temporally to predict action
            action_pred = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1], axis=0) / np.sum(self.action_buffer_mask[:, 0], axis=0)
            action_pred = torch.from_numpy(action_pred)

            # Make gripper action either -1 or 1
            if self.is_gripper_binary:
                action_pred[-1] = 1 if action_pred[-1] > 0 else -1
            
            if self.pred_discrete_arm_action:
                arm_action_pred = action_pred[:-1]
                arm_action_pred = arm_action_pred.reshape(-1, 3)
                arm_action_pred = arm_action_pred.softmax(dim=-1).argmax(dim=-1)
                action_pred = torch.cat([arm_action_pred, action_pred[-1:]], dim=-1)
            
            action_pred = action_pred.reshape(1, self.act_dim)


        self.rollout_step_counter += 1
        return action_pred


