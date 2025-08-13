"""Latent Motion Tokenizer model."""
import torch
import torch.nn.functional as F
from torch import nn
import lpips
from einops import rearrange
from transformers import ViTMAEModel
from PIL import Image
from torchvision import transforms as T
import time
from collections import OrderedDict


class LatentMotionTokenizer(nn.Module):
    def __init__(
            self,
            image_encoder,
            m_former,
            vector_quantizer,
            decoder,
            hidden_state_decoder=None,
            codebook_dim=32,
            commit_loss_w=1.,
            recon_loss_w=1.,
            recon_hidden_loss_w=1.,
            perceptual_loss_w=1.,
            use_abs_recons_loss=False,
    ):
        super().__init__()

        codebook_embed_dim = codebook_dim
        decoder_hidden_size = decoder.config.hidden_size
        m_former_hidden_size = m_former.config.hidden_size

        if isinstance(image_encoder, ViTMAEModel):
            image_encoder.config.mask_ratio = 0.0

        self.image_encoder = image_encoder.requires_grad_(False).eval()

        self.m_former = m_former

        self.vector_quantizer = vector_quantizer
        self.vq_down_resampler = nn.Sequential(
            nn.Linear(m_former_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size, codebook_embed_dim)
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, decoder_hidden_size)
        )

        self.decoder = decoder
        self.hidden_state_decoder = hidden_state_decoder

        self.commit_loss_w = commit_loss_w
        self.recon_loss_w = recon_loss_w
        self.recon_hidden_loss_w = recon_hidden_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()
        self.use_abs_recons_loss = use_abs_recons_loss

    @property
    def device(self):
        return next(self.parameters()).device


    def get_state_dict_to_save(self):
        modules_to_exclude = ['loss_fn_lpips', 'image_encoder']
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict


    @torch.no_grad()
    def decode_image(self, cond_pixel_values, given_motion_token_ids):
        quant = self.vector_quantizer.get_codebook_entry(given_motion_token_ids)
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(cond_input=cond_pixel_values, latent_motion_tokens=latent_motion_tokens_up)
        return  {
            "recons_pixel_values": recons_pixel_values,
        }


    @torch.no_grad()
    def embed(self, cond_pixel_values, target_pixel_values, pool=False, before_vq=False, avg=False):
        quant, *_ = self.tokenize(cond_pixel_values, target_pixel_values, before_vq=before_vq)
        if pool:
            latent_motion_tokens_up = self.vq_up_resampler(quant)
            flat_latent_motion_tokens_up = latent_motion_tokens_up.reshape(latent_motion_tokens_up.shape[0], -1)
            pooled_embeddings = self.decoder.transformer.embeddings.query_pooling_layer(flat_latent_motion_tokens_up)
            return pooled_embeddings
        elif avg:
            return quant.mean(dim=1)
        else:
            return quant.reshape(quant.shape[0], -1)

    def tokenize(self, cond_pixel_values, target_pixel_values, before_vq=False):
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state

        query_num = self.m_former.query_num
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]

        if before_vq:
            return latent_motion_tokens, None, None
        else:
            latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
            quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
            return quant, indices, commit_loss


    def forward(self, cond_pixel_values, target_pixel_values,
                return_recons_only=False, 
                return_motion_token_ids_only=False): 

        # Tokenization
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state
        # print("cond hidden states shape: ", cond_hidden_states.shape)
        # print("target hidden states shape: ", target_hidden_states.shape)
        
        query_num = self.m_former.query_num
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]

        # print("tokenized latent motion tokens shape: ", latent_motion_tokens.shape)

        latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
        # print("latent motion tokens down shape: ", latent_motion_tokens_down.shape)
        quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
        
        # print("quant shape: ", quant.shape)
        # quant, indices, commit_loss = self.tokenize(cond_pixel_values, target_pixel_values)

        if return_motion_token_ids_only:
            return indices # (bs, motion_query_num)

        # Detokenization
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        # print("latent motion tokens up shape: ", latent_motion_tokens_up.shape)
        recons_pixel_values = self.decoder(
            cond_input=cond_pixel_values,
            latent_motion_tokens=latent_motion_tokens_up
        )
        # print("recons pixel values shape: ", recons_pixel_values.shape)
            
        if return_recons_only:
            return {
                "recons_pixel_values": recons_pixel_values,
                "indices": indices
            }

        if self.hidden_state_decoder is not None:
            recons_hidden_states = self.hidden_state_decoder(
                cond_input = cond_hidden_states,
                latent_motion_tokens=latent_motion_tokens_up
            )

        # Compute loss
        outputs = {
            "loss": torch.zeros_like(commit_loss),
            "commit_loss": commit_loss,
            "recons_loss": torch.zeros_like(commit_loss),
            "recons_hidden_loss": torch.zeros_like(commit_loss),
            "perceptual_loss": torch.zeros_like(commit_loss)
        }

        if self.use_abs_recons_loss:
            recons_loss = torch.abs(recons_pixel_values - target_pixel_values).mean()
        else:
            recons_loss = F.mse_loss(target_pixel_values, recons_pixel_values)
        outputs["recons_loss"] = recons_loss

        if self.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss = self.loss_fn_lpips.forward(
                    target_pixel_values, recons_pixel_values, normalize=True).mean()
        else:
            perceptual_loss = torch.zeros_like(recons_loss)
        outputs["perceptual_loss"] = perceptual_loss

        loss =  self.commit_loss_w * outputs["commit_loss"] + self.recon_loss_w * outputs["recons_loss"] + \
                self.perceptual_loss_w * outputs["perceptual_loss"]
        
        if self.hidden_state_decoder is not None:
            recon_hidden_loss = F.mse_loss(target_hidden_states, recons_hidden_states)
            outputs['recons_hidden_loss'] = recon_hidden_loss
            loss += self.recon_hidden_loss_w * outputs['recons_hidden_loss']

        outputs["loss"] = loss

        # active_code_num = torch.tensor(len(set(indices.long().reshape(-1).cpu().numpy().tolist()))).float().to(loss.device)
        active_code_num = torch.tensor(torch.unique(indices).shape[0]).float().to(loss.device)
        outputs["active_code_num"] = active_code_num

        return outputs
