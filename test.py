import os
import torch
from PIL import Image
import json
import argparse
from tqdm import tqdm
import glob

from vislinginstruct.models import load_model_and_preprocess


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--name", type=str, default="my_mmlm_flant5", help="")
    # parser.add_argument("--model_type", type=str, default="eval_flant5", help="")
    parser.add_argument("--name", type=str, default="my_mmlm_vicuna", help="")
    parser.add_argument("--model_type", type=str, default="eval_vicuna7b", help="")
    args = parser.parse_args()
    return args


if  __name__ == '__main__':
    args = parse_args()

    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # load sample image
    raw_image = Image.open("/root/paddlejob/workspace/env_run/BLIVA-main/Confusing-Pictures.jpg").convert("RGB")
    # loads InstructBLIP model
    with torch.no_grad():
        model, vis_processors, _ = load_model_and_preprocess(name=args.name, 
                                                            model_type=args.model_type, 
                                                            is_eval=True, 
                                                            device=device)
    # prepare the image
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    model.eval()
    res = model.generate({"image": image, "prompt": "What is unusual about this image?"})
    print(res)

    ias = model.calculate_ias({"image": image, 
                            "prompt": "What is unusual about this image?", 
                            "text_input": "Based on the image given, the most appropriate instruction should be:"})
    print(ias)
