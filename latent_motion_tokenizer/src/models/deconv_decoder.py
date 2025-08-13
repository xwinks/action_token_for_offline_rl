import torch
import torch.nn as nn
import numpy as np
import math
def silu(x):
    return x*torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)
    
def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='replicate'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h

class ImageDecoder(nn.Module):
    def __init__(self, in_dim, n_hiddens, n_times_upsample, use_14_patch_decoder=True,norm_type='group', padding_type='replicate'):
        super().__init__()
        # everytime upsample with 2 times
        self.in_dim = in_dim
        self.n_hiddens = n_hiddens
        self.n_times_upsample = n_times_upsample
        self.norm_type = norm_type
        
        in_channels = n_hiddens * (2 ** n_times_upsample)
        
        self.layers = nn.ModuleList()
        self.final_block = nn.Sequential(
            Normalize(in_dim, norm_type),
            SiLU()
        )
        self.first_conv = nn.Conv2d(self.in_dim, in_channels, kernel_size=3, padding=1, padding_mode=padding_type)
        
        for k in range(n_times_upsample):
            in_channels = in_channels if k == 0 else n_hiddens * (2 ** (n_times_upsample - k + 1))
            out_dim = n_hiddens * (2 ** (n_times_upsample - k))
            padding = 2 if (use_14_patch_decoder and k == 0) else 0
            # kernel_size = 3 if (use_14_patch_decoder and k == 0) else 2
            # padding = 0
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels,out_channels=out_dim, kernel_size=2, stride=2, padding=padding),
                    ResBlock(out_dim, out_dim, norm_type=norm_type),
                    ResBlock(out_dim, out_dim, norm_type=norm_type),
                )
            )
    
        self.final_conv = nn.Conv2d(out_dim, out_channels=3, kernel_size=3, padding=1, padding_mode=padding_type)
    
    def forward(self, x):
        x = self.first_conv(x)
        for layer in self.layers:
            x = layer(x)
            # print("the x shape is: ", x.shape)
        x = self.final_conv(x)
        return x
    
    
class MotionGuidedImageDecoder(nn.Module):
    def __init__(self, in_dim, n_hiddens, n_times_upsample, motion_dim, n_motion_tokens,use_14_patch_decoder=True, image_encoder = None,
                 langauge_dim = None,norm_type='group', padding_type='replicate'):
        super().__init__()
        
        self.motion_dim = motion_dim
        self.image_encoder = image_encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, motion_dim * 2),
            SiLU(),
            nn.Linear(motion_dim * 2, motion_dim * 2)
        )
        self.motion_dim = motion_dim
        
        if langauge_dim is not None:
            self.language_proj = nn.Linear(langauge_dim, motion_dim * 2)
            attn_dim = motion_dim * 4
            self.vision_proj = nn.Linear(in_dim, motion_dim * 4)
            self.attention = nn.MultiheadAttention(embed_dim=motion_dim * 4, num_heads=2)

        else:
            attn_dim = motion_dim * 2
            self.vision_proj = nn.Linear(in_dim, motion_dim * 2)
            self.attention = nn.MultiheadAttention(embed_dim=motion_dim * 2, num_heads=2, batch_first=True)

        
        self.image_decoder = ImageDecoder(in_dim + attn_dim, n_hiddens, 
                n_times_upsample, norm_type = norm_type, padding_type = padding_type, use_14_patch_decoder = use_14_patch_decoder)
        
    
    def forward(self, cond_input, latent_motion_tokens, language_features = None):
        with torch.no_grad():
            vision_tokens = self.image_encoder(cond_input).last_hidden_state
        motion_tokens = self.motion_proj(latent_motion_tokens)
        B,N,D = motion_tokens.shape
        if language_features is not None:
            language_features = self.language_proj(language_features)
            language_features = language_features.unsqueeze(1).expand(-1, N, -1)
            attn_value = torch.cat([language_features, motion_tokens], dim=-1)
        else:
            attn_value = motion_tokens
            
        vision_attn_input = self.vision_proj(vision_tokens)
        vision_attn_output = self.attention(vision_attn_input, attn_value, attn_value)[0]
        B,N_Tokens,D = vision_tokens.shape
        vision_attn_output = vision_attn_output.view(B, N_Tokens, -1)
        
        # print("the attn query shape is: ", attn_value.shape)
        # print("the vision attn input shape is: ", vision_attn_input.shape)
        # print("the vision attn output shape is: ", vision_attn_output.shape)
        
        x = torch.cat([vision_tokens, vision_attn_output], dim=-1)
        # print("the x shape is: ", x.shape)
        # print("the vision tokens shape is: ", vision_tokens.shape)
        # print("the vision attn output shape is: ", vision_attn_output.shape)
        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, int(math.sqrt(N_Tokens)), int(math.sqrt(N_Tokens)))
        x = self.image_decoder(x)
        return x
        

if __name__ == "__main__":
    # conv_model = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, padding_mode='replicate')
    # conv_model = ResBlock(in_channels=512, out_channels=512, conv_shortcut=True, dropout=0.0, norm_type='group', padding_type='replicate')
    # print(conv_model(torch.randn(1, 512, 10, 10)).shape)
    
    # convtranspose_model = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
    # a = convtranspose_model(torch.randn(1, 512, 14, 14))
    # print("the a shape is: ", a.shape)
    # b = convtranspose_model(a)
    # print("the b shape is: ", b.shape)
    
    input_image_tokens = torch.randn(1, 2048, 16, 16)
    
    decoder = ImageDecoder(in_dim=2048, n_hiddens=128, n_times_upsample=4, norm_type='group', padding_type='replicate')
    output = decoder(input_image_tokens)
    print("the output shape is: ", output.shape)
    
    
    # attn_input = torch.randn(16, 16,64)
    # attn_output = torch.randn(16, 1,64)
    # attention_module = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
    # output = attention_module(attn_input, attn_output, attn_output)
    # print("the output shape is: ", output[0].shape)
    # print("the output is: ", output[1].shape)
    
    # model = nn.ConvTranspose2d(256,256, kernel_size=2, stride=2, padding=1)
    # a = model(torch.randn(1, 256, 8, 8))
    # print("the a shape is: ", a.shape)
    # b = model(a)
    # print("the b shape is: ", b.shape)