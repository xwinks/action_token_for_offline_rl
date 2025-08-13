import pyrootutils
import os
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
from omegaconf import OmegaConf
import hydra
import torchvision.transforms as T
import json
import argparse
from tqdm import tqdm
from glob import glob
import cv2
from PIL import Image, ImageFont, ImageDraw
import torch
import math
from functools import partial
from transformers import AutoTokenizer
from transformers.utils import FEATURE_EXTRACTOR_NAME, get_file_from_repo
import numpy as np
from common.models.model_utils import load_model

def get_image_processor(vision_processor_config):
    input_size = (vision_processor_config['size'], vision_processor_config['size'])
    rgb_mean = vision_processor_config['image_mean']
    rgb_std = vision_processor_config['image_std']
    image_transform = T.Compose([
        T.ToTensor(),
        T.Resize(input_size, interpolation=Image.BICUBIC),
        T.Normalize(rgb_mean, rgb_std)])
    return image_transform

def get_image_seq_post_processor(image_seq, image_std, image_mean):
    image_mean = torch.tensor(image_mean).view(1,3,1,1)
    image_std = torch.tensor(image_std).view(1,3,1,1)
    image_seq = image_seq * image_std + image_mean # (t, c, h, w)
    image_seq = torch.clamp(image_seq, min=0, max=1)
    image_seq = list(map(T.ToPILImage(), image_seq.unbind(dim=0)))
    return image_seq


def visualization(
        lang_goal,
        orig_video, 
        decoding_mode2preds,
        path, 
        image_seq_post_processor
    ):
    n_cols, c, h, w = orig_video.shape
    n_rows = len(decoding_mode2preds)+1
    h = h + 30

    orig_video = image_seq_post_processor(orig_video)
    initial_frame = orig_video[0]
    gt_subsequent_frames = orig_video[1:]
    for decoding_mode, preds in decoding_mode2preds.items():
        preds['frame_preds'] = image_seq_post_processor(preds['frame_preds'])
        preds['latent_motion_id_preds'] = preds['latent_motion_id_preds'].numpy().tolist()

    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=12)
    compare_img = Image.new('RGB', size=(n_cols*w, n_rows*h))
    draw_compare_img = ImageDraw.Draw(compare_img)
    
    for i in range(n_rows):
        compare_img.paste(initial_frame, box=(0, i*h))

    for j in range(n_cols-1):
        compare_img.paste(gt_subsequent_frames[j], box=((j+1)*w, 0))

        for i, (decoding_mode, preds) in enumerate(decoding_mode2preds.items()):
            compare_img.paste(preds['frame_preds'][j], box=((j+1)*w, (i+1)*h))
            draw_compare_img.text(((j+1)*w, (i+2)*h-20), f"{preds['latent_motion_id_preds'][j]}", font=font, fill=(0, 255, 0))
            
            if j == 0:
                draw_compare_img.text((0, (i+2)*h-20), f"{decoding_mode}", font=font, fill=(0, 255, 0))

    draw_compare_img.text((0, h-20), f"{lang_goal}", font=font, fill=(0, 255, 0))
    compare_img.save(f"{path}-{'_'.join(lang_goal.split())}.png")


    h = h - 30
    fps = 4
    for i, (decoding_mode, preds) in enumerate(decoding_mode2preds.items()):
        output_video_path = f"{path}-{'_'.join(lang_goal.split())}-{decoding_mode}.mp4"
        images = preds['frame_preds']
        images = [initial_frame] + images

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

        for image in images:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            video_writer.write(image_cv)
        video_writer.release()

def inference(
        moto_gpt, 
        latent_motion_tokenizer, 
        lang_tokenizer, 
        image_processor, 
        image_seq_post_processor,
        num_gen_frames,
        delta_t,
        moto_gpt_seq_len,
        input_dir, 
        output_dir
    ):

    device = moto_gpt.device
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(input_dir, "lang_annotations.json")) as f:
        lang_annotations = json.load(f)

    video_dir = os.path.join(input_dir, "videos")
    for video_path in tqdm(glob(os.path.join(video_dir, "*.mp4"))):
        video_basename = os.path.basename(video_path)
        video = cv2.VideoCapture(video_path)
        video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(0, min(video_len, num_gen_frames*delta_t+1), delta_t):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = image_processor(Image.fromarray(frame).convert("RGB"))
            frames.append(frame)

        video.release()
        frames = torch.stack(frames).to(device) # (f, c, h, w)
        initial_frame = frames[0]
        subsequent_frames = frames[1:]
        exact_num_gen_frames = subsequent_frames.shape[0]

        gt_latent_motion_ids = latent_motion_tokenizer(
            cond_pixel_values=frames[:-1],
            target_pixel_values=frames[1:],
            return_motion_token_ids_only=True
        )

        recons_subsequent_frames = latent_motion_tokenizer.decode_image(
            cond_pixel_values=frames[:-1],
            given_motion_token_ids=gt_latent_motion_ids
        )["recons_pixel_values"] # (exact_num_gen_frames, c, h, w)


        decoding_mode2preds = {
            "ground_truth_recons": {
                "frame_preds": recons_subsequent_frames.detach().cpu(),
                "latent_motion_id_preds": gt_latent_motion_ids.detach().cpu()
            }
        }

        decoding_mode2latent_motion_decoding_kwargs = {
            "sampleFalse_beam1": {
                "temperature": 1.0, 
                "sample": False, 
                "top_k": 0, 
                "top_p": 1.0,
                "beam_size": 1, 
                "parallel": False
            },

            "sampleTrue_beam5": {
                "temperature": 1.0, 
                "sample": True, 
                "top_k": 0, 
                "top_p": 1.0,
                "beam_size": 5, 
                "parallel": False
            },

            "sampleFalse_beam5": {
                "temperature": 1.0, 
                "sample": False, 
                "top_k": 0, 
                "top_p": 1.0,
                "beam_size": 5, 
                "parallel": False
            },
        }

        
        lang_goal = lang_annotations[video_basename]
        lang_inputs = lang_tokenizer(lang_goal, return_tensors='pt', padding=True)
        tokenized_text = lang_inputs.input_ids.to(device)
        lang_attention_mask = lang_inputs.attention_mask.to(device)

        attention_mask = torch.ones(1, moto_gpt_seq_len).long().to(device)
        dummy_latent_motion_ids = torch.zeros((1, moto_gpt_seq_len, gt_latent_motion_ids.shape[-1])).long().to(device)
        latent_mask = attention_mask

        for decoding_mode, latent_motion_decoding_kwargs in decoding_mode2latent_motion_decoding_kwargs.items():
            gen_iter_num = math.ceil(num_gen_frames / moto_gpt_seq_len)

            frame_preds = []
            latent_motion_id_preds = []

            cur_cond_pixel_values = initial_frame.unsqueeze(0) # (b, c, h, w)
            cur_latent_motion_ids = dummy_latent_motion_ids.clone() # (b, moto_gpt_seq_len, per_latent_motion_len)

            cur_initial_frame = initial_frame.unsqueeze(0).unsqueeze(0) # (b, 1, c, h, w)

            
            for _ in range(gen_iter_num):
                for buffer_len in range(1, moto_gpt_seq_len+1):
                    pred = moto_gpt(
                        rgb=cur_initial_frame, 
                        language=tokenized_text,
                        attention_mask=attention_mask,
                        latent_motion_ids=cur_latent_motion_ids,
                        latent_mask=latent_mask,
                        train=False,
                        lang_attention_mask=lang_attention_mask,
                        buffer_len=buffer_len,
                        **latent_motion_decoding_kwargs,
                    )
                    cur_latent_motion_id_preds = pred['latent_motion_id_preds'] # (b, per_latent_motion_len)
                    cur_latent_motion_ids[:,buffer_len-1] = cur_latent_motion_id_preds
                    cur_frame_preds = latent_motion_tokenizer.decode_image(
                        cond_pixel_values=cur_cond_pixel_values,
                        given_motion_token_ids=cur_latent_motion_id_preds.reshape(-1, cur_latent_motion_id_preds.shape[-1]) # (b, per_latent_motion_len)
                    )["recons_pixel_values"] # (b, c, h, w)
                    cur_cond_pixel_values = cur_frame_preds
                    frame_preds.append(cur_frame_preds.detach().cpu())
                    latent_motion_id_preds.append(cur_latent_motion_id_preds.detach().cpu())

                cur_initial_frame = cur_frame_preds.unsqueeze(1) # (b, 1, c, h, w)

            frame_preds = torch.cat(frame_preds, dim=0)[:exact_num_gen_frames]
            latent_motion_id_preds = torch.cat(latent_motion_id_preds, dim=0)[:exact_num_gen_frames]
            decoding_mode2preds[decoding_mode] = {
                "frame_preds": frame_preds.detach().cpu(),
                "latent_motion_id_preds": latent_motion_id_preds.detach().cpu()
            }

        basename = os.path.basename(video_path).split(".")[0]
        visualization(
            lang_goal=lang_goal,
            orig_video=frames.detach().cpu(), 
            decoding_mode2preds=decoding_mode2preds,
            image_seq_post_processor=image_seq_post_processor,
            path=os.path.join(output_dir, basename)
        )



def main(args):
    # Prepare Moto-GPT
    print(f"loading Moto-GPT from {args.moto_gpt_path} ...")
    moto_gpt = load_model(args.moto_gpt_path)
    moto_gpt_config = moto_gpt.config
    moto_gpt = moto_gpt.cuda()
    moto_gpt.eval()

    # Prepare tokenizers and processors
    lang_tokenizer = AutoTokenizer.from_pretrained(moto_gpt_config['model_lang']['pretrained_model_name_or_path'])
    vision_processor_config = json.load(open(get_file_from_repo(moto_gpt_config['model_vision']['pretrained_model_name_or_path'], FEATURE_EXTRACTOR_NAME)))
    image_processor = get_image_processor(vision_processor_config)
    image_seq_post_processor = partial(
        get_image_seq_post_processor, 
        image_std=vision_processor_config['image_std'], 
        image_mean=vision_processor_config['image_mean']
    )

    # Prepare Latent Motion Tokenizer
    print(f"loading Latent Motion Tokenizer from {args.latent_motion_tokenizer_path} ...")
    latent_motion_tokenizer = load_model(args.latent_motion_tokenizer_path)
    latent_motion_tokenizer = latent_motion_tokenizer.cuda()
    latent_motion_tokenizer.eval()


    # Run inference
    inference(
        moto_gpt=moto_gpt, 
        latent_motion_tokenizer=latent_motion_tokenizer, 
        lang_tokenizer=lang_tokenizer, 
        image_processor=image_processor, 
        image_seq_post_processor=image_seq_post_processor,
        num_gen_frames=args.num_gen_frames,
        delta_t=args.delta_t,
        moto_gpt_seq_len=moto_gpt_config['sequence_length'],
        input_dir=args.input_dir, 
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moto_gpt_path', type=str, required=True)
    parser.add_argument('--latent_motion_tokenizer_path', type=str, required=True)
    parser.add_argument('--num_gen_frames', type=int, default=4)
    parser.add_argument('--delta_t', type=int, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    main(args)

    