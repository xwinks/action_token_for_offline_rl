import torch
from transformers import T5Tokenizer, T5EncoderModel

class T5TextEncoder:
    def __init__(self, model_name="t5-large", device=None):
        """
        封装T5编码器和tokenizer
        Args:
            model_name (str): 预训练模型名称
            device (str or torch.device): 设备, 如"cuda"或"cpu"
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, text, return_cls=True, padding=True, truncation=True, max_length=None):
        """
        编码文本为T5的表示
        Args:
            text (str or list): 输入文本
            return_cls (bool): 是否返回第一个token的表示（类似[CLS]）
            padding (bool): 是否padding
            truncation (bool): 是否truncation
            max_length (int or None): 最大长度
        Returns:
            torch.Tensor: (batch_size, seq_len, hidden_size) 或 (batch_size, hidden_size)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            encoder_outputs = self.model(**inputs)
        encoded_representation = encoder_outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        if return_cls:
            return encoded_representation[:, 0, :]  # (batch, hidden_size)
        else:
            return encoded_representation  # (batch, seq_len, hidden_size)

# --------- CLIP Text Encoder 版本 ---------
import clip


class CLIPTextEncoder:
    def __init__(self, model_name="ViT-B/32", device=None):
        """
        封装CLIP文本编码器和tokenizer
        Args:
            model_name (str): CLIP预训练模型名称, 如"ViT-B/32"
            device (str or torch.device): 设备, 如"cuda"或"cpu"
        """
        if clip is None:
            raise ImportError("clip package not found. Please install openai-clip to use CLIPTextEncoder.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        self.model.eval()

    def encode(self, text, return_cls=True):
        """
        编码文本为CLIP的表示
        Args:
            text (str or list): 输入文本
            return_cls (bool): 是否返回全局文本embedding（True时返回全局embedding，False时返回token级embedding）
        Returns:
            torch.Tensor: (batch_size, hidden_size) 或 (batch_size, seq_len, hidden_size)
        """
        if isinstance(text, str):
            text = [text]
        with torch.no_grad():
            text_tokens = clip.tokenize(text, truncate=True).to(self.device)
            if return_cls:
                # 返回全局文本embedding
                text_features = self.model.encode_text(text_tokens)  # (batch, hidden_size)
                return text_features
            else:
                # 返回token级embedding（取transformer输出）
                x = self.model.token_embedding(text_tokens)  # (batch, seq_len, dim)
                x = x + self.model.positional_embedding
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.model.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                # x: (batch, seq_len, dim)
                return x

if __name__ == "__main__":
    encoder = T5TextEncoder(model_name="t5-large")
    text = "Hello, how are you?"
    # 获取所有token的表示
    all_tokens = encoder.encode(text, return_cls=False)
    print("all_tokens shape:", all_tokens.shape)  # (1, T, hidden_size)
    # 获取第一个token的表示
    cls_token = encoder.encode(text, return_cls=True)
    print("cls_token shape:", cls_token.shape)    # (1, hidden_size)


    encoder = CLIPTextEncoder(model_name="ViT-B/32")
    text = "Hello, how are you?"
    # 获取所有token的表示
    # all_tokens = encoder.encode(text, return_cls=False)
    # print("all_tokens shape:", all_tokens.shape)  # (1, T, hidden_size)
    # 获取第一个token的表示
    cls_token = encoder.encode(text, return_cls=True)
    print("cls_token shape:", cls_token.shape)    # (1, hidden_size)