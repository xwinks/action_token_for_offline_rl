import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.sys.path.append("../")
from latent_motion_tokenizer.src.models.timm_dinov2_model import TimmDinoV2VsionEncoder
import torch

if __name__ == "__main__":
    vision_encoder = TimmDinoV2VsionEncoder()
    image = torch.randn(1, 3, 224, 224)
    output = vision_encoder(image)
    print(output.last_hidden_state.shape)