import numpy as np
import torch
import torch.nn.functional as F
from utils import to_numpy

def top_k_acc(knn_labels, gt_labels, k):
    accuracy_per_sample = torch.any(knn_labels[:, :k] == gt_labels, dim=1).float()
    return torch.mean(accuracy_per_sample)

class Evaluator(object):


    def __init__(self, model, n=2):

        self.model = model

        self.extract_device = 'cuda'
        self.eval_device = 'cuda'

        self.num_classes = self.model.num_classes
        self.embed_dim = self.model.embed_dim
        self.n = n

    @torch.no_grad()
    def extract(self, dataloader):
  
        self.model.eval()
        self.model.to(self.extract_device)

        num_collections = len(dataloader.dataset)
        num_total_images = num_collections * self.n

        results_dict = {'single': {'logits': np.zeros((num_total_images, self.num_classes)),
                                  'classes': np.zeros(num_total_images),
                                  'paths': []},
                       'mv_collection': {'logits': np.zeros((num_collections, self.num_classes)),
                                    'classes': np.zeros(num_collections),
                                    'paths': []}}

        if self.n == 1:
            del results_dict['mv_collection']

        s = 0
        for i, data in enumerate(dataloader):
            images, targets, paths = data
            images = images.to(self.extract_device)
            batch_output = self.model(images)
            e = s + len(images)
            for view_type in batch_output:
                if view_type == 'single':
                    multiplier = self.n
                    t = targets.view(len(images) * self.n)
                    p = np.array(paths).T.flatten().tolist()
                else:
                    multiplier = 1
                    t = targets[:, 0]
                    p = []
                    for w in range(len(paths[0])):
                        l = []
                        for j in range(self.n):
                            l.append(paths[j][w])
                        p.append(l)

                results_dict[view_type]['logits'][s * multiplier: e * multiplier] = to_numpy(batch_output[view_type]['logits'])
                results_dict[view_type]['classes'][s * multiplier: e * multiplier] = to_numpy(t)
                results_dict[view_type]['paths'].extend(p)
            s = e

        results_dict['single']['paths'] = np.array(results_dict['single']['paths'])
        duplicates = set()
        kept_indices = []
        for i in range(len(results_dict['single']['paths'])):
            p = results_dict['single']['paths'][i]
            if p not in duplicates:
                duplicates.add(p)
                kept_indices.append(i)

        for key in results_dict['single']:
            results_dict['single'][key] = results_dict['single'][key][kept_indices]

        for view_type in results_dict:
            results_dict[view_type]['logits'] = torch.from_numpy(results_dict[view_type]['logits']).squeeze()
            results_dict[view_type]['classes'] = results_dict[view_type]['classes'].squeeze()

        return results_dict

    def get_metrics(self, logits, gt_labels):

        try:
            logits = logits
            gt_labels = gt_labels
            predictions = torch.argsort(logits, dim=1, descending=True)
        except torch.cuda.OutOfMemoryError:
            logits = logits.cpu()
            gt_labels = gt_labels.cpu()
            predictions = torch.argsort(logits, dim=1, descending=True)

        class_1 = top_k_acc(predictions, gt_labels, 1)
        class_2 = top_k_acc(predictions, gt_labels, 2)
        class_5 = top_k_acc(predictions, gt_labels, 5)
        class_10 = top_k_acc(predictions, gt_labels, 10)
        class_100 = top_k_acc(predictions, gt_labels, 100)

        metrics = {
            'top1_acc': class_1.item(),
            'top2_acc': class_2.item(),
            'top5_acc': class_5.item(),
            'top10_acc': class_10.item(),
            'top100_acc': class_100.item(),
        }
        return metrics

    def evaluate(self, dataloader):

        results_dict = self.extract(dataloader)
        metrics_dict = {}

        for view_type in results_dict:
            gt_classes = np.expand_dims(results_dict[view_type]['classes'], -1)
            metrics = self.get_metrics(results_dict[view_type]['logits'].to(self.eval_device), torch.from_numpy(gt_classes).to(self.eval_device))
            metrics_dict[view_type] = metrics

        return metrics_dict
