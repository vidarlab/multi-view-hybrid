import hashlib
import json
import os

import torch


def torch_safe_load(module, state_dict, strict=True):
    module.load_state_dict({
        k.replace('module.', ''): v for k, v in state_dict.items()
    }, strict=strict)


class EngineBase(object):
    def __init__(self, model, optimizer_model, criterion, lr_scheduler_model, evaluator, 
                 save_dir=None, md_loss=None, grad_clip_norm=None, logger=None):


        self.device = 'cuda'
        self.model = model
        self.optimizer_model = optimizer_model
        self.criterion = criterion
        self.lr_scheduler_model = lr_scheduler_model
        self.save_dir = save_dir
        self.evaluator = evaluator
        self.md_loss = md_loss
        self.grad_clip = grad_clip_norm
        self.metadata = {}
        self.logger = logger

    def model_to_device(self):
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, val_loader):
        if self.evaluator is None:
            self.logger.info('[Evaluate] Warning, no evaluator is defined. Skip evaluation')
            return
        scores = self.evaluator.evaluate(val_loader)
        return scores

    def save_models(self, save_to, metadata=None):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer_model': self.optimizer_model.state_dict(),
        }
        if self.lr_scheduler_model is not None:
            state_dict['lr_scheduler_model'] = self.lr_scheduler_model.state_dict()
        print('Saving model to {}'.format(save_to))
        torch.save(state_dict, save_to)
        self.logger.info('state dict is saved to {}'.format(save_to))

    def load_models(self, state_dict_path, load_keys=None):
        with open(state_dict_path, 'rb') as fin:
            model_hash = hashlib.sha1(fin.read()).hexdigest()
            self.metadata['pretrain_hash'] = model_hash

        state_dict = torch.load(state_dict_path, map_location='cpu')

        if 'model' not in state_dict:
            torch_safe_load(self.model, state_dict, strict=False)
            return

        if not load_keys:
            load_keys = ['model', 'optimizer_model', 'lr_scheduler_model']
        
        for key in load_keys:
            try:
                torch_safe_load(getattr(self, key), state_dict[key])
            except RuntimeError as e:
                print('Unable to import state_dict, missing keys are found. {}'.format(e))
                torch_safe_load(getattr(self, key), state_dict[key], strict=False)
        print('state dict is loaded from {} (hash: {}), load_key ({})'.format(state_dict_path, model_hash, load_keys))
