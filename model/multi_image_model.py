import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class MultiImageHybrid(nn.Module):

    def __init__(self, arch, num_classes, n,  pretrained_weights=True):
        
        super().__init__()

        self.n = n
        self.num_classes = num_classes
        self.pretrained_weights = pretrained_weights

        drop_rate = .0 if 'tiny' in arch else .1
        self.model = timm.create_model(arch, pretrained=self.pretrained_weights, num_classes=self.num_classes, drop_rate=drop_rate)
        for block in self.model.blocks:
            block.attn.fused_attn = False
        
        self.embed_dim = self.model.embed_dim

        for block in self.model.blocks:
            block.attn.proj_drop = nn.Dropout(p=0.0)

        self.img_embed_matrix = nn.Parameter(torch.zeros(1, n, self.embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.img_embed_matrix)

        nn.init.zeros_(self.model.head.weight)
        nn.init.zeros_(self.model.head.bias)

    def format_multi_image_tokens(self, x, batch_size, tokens_per_image):

        x = einops.rearrange(x, '(b n) s c -> b (n s) c', b=batch_size, n=self.n)
        first_img_token_idx = 0
        if self.model.cls_token is not None:
            # Need to remove all excess CLS tokens
            for i in range(1, self.n):
                excess_cls_index = i * tokens_per_image + 1
                x = torch.cat((x[:, :excess_cls_index], x[:, excess_cls_index + 1:]), dim=1)
            first_img_token_idx = 1

        image_embeddings = F.normalize(self.img_embed_matrix, dim=-1)
        x[:, first_img_token_idx:] += torch.repeat_interleave(image_embeddings, tokens_per_image, dim=1)
        return x

    def forward(self, x):

        batch_size = len(x)
        output_dict = {'single': {}}
        if self.n > 1:
            output_dict['mv_collection'] = {}

        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.model.patch_embed(x)

        tokens_per_image = x.shape[1]
        x = self.model._pos_embed(x)

        for view_type in output_dict:

            tokens = x.clone()
            if view_type == 'mv_collection':
                tokens = self.format_multi_image_tokens(tokens, batch_size, tokens_per_image)
            tokens = self.model.blocks(tokens)
            tokens = self.model.norm(tokens)
            output_dict[view_type]['logits'] = self.model.forward_head(tokens)

        return output_dict