import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.gelu(self.norm(self.fc1(x)))
        x = self.fc2(x)
        return x

class CrossAttnBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, mlp_ratio=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x, mem_key_value):
        x = self.norm1(x)
        mem_key_value = self.norm1(mem_key_value)
        q, _ = self.attn(query = x, key = mem_key_value, value = mem_key_value)
        x = x + q
        x = x + self.norm2(self.ffn(x))
        return x

class QueryHead(nn.Module):
    def __init__(self, hidden_size, n_heads = 12, depth = 6):
        super().__init__()
        self.layers = nn.ModuleList([CrossAttnBlock(hidden_size, n_heads) for _ in range(depth)])

        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, latent_motion_embeddings, mem_key_value):
        B = mem_key_value.shape[0]
        latent_motion_embeddings = latent_motion_embeddings.expand( B, -1, -1)
        for layer in self.layers:
            latent_motion_embeddings = layer(latent_motion_embeddings, mem_key_value)
        latent_motion_embeddings = self.output_proj(latent_motion_embeddings)
        return latent_motion_embeddings


class MotoGPTActionPrediction(nn.Module):
    def __init__(
            self,
            model_lang,
            model_vision,
            model_causal_transformer,
            num_patch,
            act_dim,
            hidden_size,
            language_feat_dim,
            latent_motion_codebook_size,
            num_latent_motion,
            image_feat_dim,
            patch_feat_dim,
            action_chunking_size = 5,
            freeze_lang=True,
            freeze_vision=True,
            **kwargs
    ):
        super().__init__()
        self.model_lang = model_lang
        self.model_vision = model_vision
        self.model_causal_transformer = model_causal_transformer
        self.hidden_size = hidden_size
        self.freeze_lang = freeze_lang
        self.freeze_vision = freeze_vision
        self.language_feat_dim = language_feat_dim
        self.image_feat_dim = image_feat_dim
        self.patch_feat_dim = patch_feat_dim
        
        self.language_projection = MLP(self.language_feat_dim, self.hidden_size, self.hidden_size)
        self.image_projection = MLP(self.image_feat_dim, self.hidden_size, self.hidden_size)
        self.patch_projection = MLP(self.patch_feat_dim, self.hidden_size, self.hidden_size)
        
        self.motion_embedding = torch.nn.Parameter(torch.zeros(latent_motion_codebook_size, self.hidden_size))
        nn.init.trunc_normal_(self.motion_embedding, std=0.02)
        self.motion_position_embedding = torch.nn.Parameter(torch.zeros(num_latent_motion, self.hidden_size))
        nn.init.trunc_normal_(self.motion_position_embedding, std=0.02)
        
        self.image_position_embedding = torch.nn.Parameter(torch.zeros(1, num_patch + 1, self.hidden_size))
        nn.init.trunc_normal_(self.image_position_embedding, std=0.02)
        
        if self.freeze_lang:
            for param in self.model_lang.parameters():
                param.requires_grad = False
        if self.freeze_vision:
            for param in self.model_vision.parameters():
                param.requires_grad = False
        
        self.query_head = QueryHead(hidden_size, n_heads = 12, depth = 6)

        self.learnable_action_embedding = torch.nn.Parameter(torch.zeros(1, action_chunking_size, self.hidden_size))
        nn.init.trunc_normal_(self.learnable_action_embedding, std=0.02)
        self.action_prediction_head = MLP(self.hidden_size, self.hidden_size, act_dim)
        
    def forward(self, image, language, language_attention_mask, motion_idx):
        # print("the language is: ", language)
        if self.freeze_lang:
            with torch.no_grad():
                lang_embeddings = self.model_lang(input_ids=language, attention_mask=language_attention_mask).last_hidden_state
        else:
            lang_embeddings = self.model_lang(input_ids=language, attention_mask=language_attention_mask).last_hidden_state
            
        if self.freeze_vision:
            with torch.no_grad():
                obs_embeddings, patch_embeddings = self.model_vision(image)
        else:   
            obs_embeddings, patch_embeddings = self.model_vision(image)
        
        lang_embeddings = self.language_projection(lang_embeddings)
        obs_embeddings = self.image_projection(obs_embeddings)
        patch_embeddings = self.patch_projection(patch_embeddings)
        obs_embeddings = obs_embeddings.unsqueeze(1)
        image_embeddings = torch.cat([patch_embeddings, obs_embeddings], dim=1)
        image_embeddings = image_embeddings + self.image_position_embedding
        
        motion_embeddings = self.motion_embedding[motion_idx]
        motion_embeddings = motion_embeddings + self.motion_position_embedding
        # print("the motion_embeddings is: ", motion_embeddings.shape)
        # print("the image_embeddings is: ", image_embeddings.shape)
        motion_embeddings = motion_embeddings.squeeze(1)
        
        
        condition_input_embeddings = torch.cat([lang_embeddings, image_embeddings, motion_embeddings], dim=1)
        
        output = self.model_causal_transformer(
            inputs_embeds=condition_input_embeddings,
            attention_mask=None,
        )
        hidden_states = output.last_hidden_state
        
        action_feature = self.query_head(self.learnable_action_embedding, hidden_states)
        action_prediction = self.action_prediction_head(action_feature)
        return action_prediction
        # action_prediction = self.action_prediction_head(hidden_states)

        
    def compute_loss(self, image, batch):
        language=batch['lang_input_ids']
        language_attention_mask=batch['lang_attention_mask']
        gt_latent_motion_ids=batch['generated_latent_motion_ids']
        gt_action=batch['action']
        action_prediction = self(image, language, language_attention_mask, gt_latent_motion_ids)
        # print("the predicted_latent_motion_logits is: ", predicted_latent_motion_logits.shape)
        # print("the gt_latent_motion_ids is: ", gt_latent_motion_ids.shape)
        # print("the action_prediction is: ", action_prediction.shape)
        # print("the gt_action is: ", gt_action.shape)
        # B,T,D = gt_action.shape
        # gt_action = gt_action.reshape(B, T*D)
        # action_prediction = action_prediction.reshape(B, T*D)
        loss = F.mse_loss(action_prediction, gt_action)
        return {
            "loss": loss,
        }
    
    def get_state_dict_to_save(self):
        modules_to_exclude = ['model_lang', 'model_vision']
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict
    
    
    
    