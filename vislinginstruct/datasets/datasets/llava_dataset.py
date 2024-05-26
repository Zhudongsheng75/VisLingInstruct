
import torch

from vislinginstruct.datasets.datasets.base_dataset import BaseDataset


import os
import json
import re

from PIL import Image
import numpy as np
import torch

class LLAVA150kDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, 'COCO_train2014_' + ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        conversations = ann["conversations"]

        text_input = ''
        for dialog in conversations[:-1]:
            if dialog['from'] == 'human':
                text_input += 'Qusetion: {} \n '.format(dialog['value'].replace('<image>', '').replace('\n', ''))
            else:
                text_input += 'Answer: {} \n '.format(dialog['value'])
        text_input += 'Answer: '
        text_output = conversations[-1]['value']

        while len(text_input.split(' ')) > 400:
            indices = [s.start() for s in re.finditer('Qusetion:', text_input)]
            text_input = text_input[indices[1]:]

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
        }

    def collater(self, samples):
        image_list, question_list, answer_list = [], [], [],

        for sample in samples:
            image_list.append(sample["image"])
           
            question_list.append(sample["text_input"])

            answers = sample["text_output"]

            answer_list.append(answers)
        

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }
