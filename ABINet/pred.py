import argparse
import logging
import os
import glob
import tqdm
import torch
import PIL
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from .utils import Config, Logger, CharsetMapper

device = 'cuda'

def get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    logging.info(model)
    model = model.eval()
    return model

def preprocess(img, width, height):
    img = cv2.resize(img, (width, height))
    img = transforms.ToTensor()(img).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    return (img-mean[...,None,None]) / std[...,None,None]


def postprocess(output, charset, model_eval):
    def _get_output(last_output, model_eval):
        if isinstance(last_output, (tuple, list)):
            for res in last_output:
                if res['name'] == model_eval: output = res
        else:
            output = last_output
        return output

    def _decode(logit):
        """ Greed decode """
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, charset.max_length))  # one for end-token
        return pt_text, pt_scores, pt_lengths

    output = _get_output(output, model_eval)
    logits, pt_lengths = output['logits'], output['pt_lengths']
    pt_text, pt_scores, pt_lengths_ = _decode(logits)

    return pt_text, pt_scores, pt_lengths_

def load(model, file, device=None, strict=True):
    if device is None: device = 'cpu'
    elif isinstance(device, int): device = torch.device('cuda', device)
    assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    return model

class Predictor:
    def __init__(self, cfg, ckpt):
        self.config = Config(cfg)
        self.config.model_eval='alignment'
        self.config.global_phase = 'test'
        self.config.model_checkpoint = ckpt
        self.config.model_vision_checkpoint, self.config.model_language_checkpoint = None, None

        self.model = get_model(self.config).to(device)
        self.model = load(self.model, self.config.model_checkpoint, device=device)
        self.charset = CharsetMapper(filename=self.config.dataset_charset_path,
                                max_length=self.config.dataset_max_length + 1)

    def pred(self, img):
        img = preprocess(img, self.config.dataset_image_width, self.config.dataset_image_height)
        img = img.to(device)
        res = self.model(img)
        pt_text, _, __ = postprocess(res, self.charset, self.config.model_eval)
        return pt_text[0]
