import os
import itertools

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


def extract_ids(im_path):
    hotel_id = im_path.split('/')[-2]
    img_id = im_path.split('/')[-1].split('.')[0]
    return img_id, hotel_id


def generate_target2indices(targets):
    target2indices = {}
    for i in range(len(targets)):
        t = targets[i]
        if t not in target2indices:
            target2indices[t] = [i]
        else:
            target2indices[t].append(i)
    return target2indices


class HotelsDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, split, n=2, train=False, classes=None):

        self.all_paths = np.load(os.path.join(data_dir, f'{split}.npy')).tolist()
        for i in range(len(self.all_paths)):
            self.all_paths[i] = f'{data_dir}/' + self.all_paths[i]
        self.all_paths = np.array(self.all_paths)
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        if not classes:
            self.classes, self.class_to_idx = self.find_classes()
        else:
            # Assure using same classes indcies as generated for training dataset
            self.classes = classes
            self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        self.num_classes = len(self.class_to_idx)
        self.n = n
        self.train = train

        self.samples = self.make_dataset()
        self.image_paths = [s[0] for s in self.samples]
        self.targets = [int(s[1]) for s in self.samples]
        self.targets2indices = generate_target2indices(self.targets)

        if not self.train:
            self.samples = self.get_all_collection_combos()
            self.image_paths = [s[0] for s in self.samples]
            self.targets = [int(s[1]) for s in self.samples]

    def find_classes(self):
        classes = set()
        for path in self.all_paths:
            _, hotel_id = extract_ids(path)
            classes.add(hotel_id)
        classes = list(classes)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self):
        samples = []
        for path in self.all_paths:
            _, hotel_id = extract_ids(path)
            if hotel_id in self.class_to_idx:
                item = (path, self.class_to_idx[hotel_id])
                samples.append(item)
        return samples

    def get_all_collection_combos(self):
        samples = []
        for t, indices in self.targets2indices.items():
            paths = [self.image_paths[i] for i in indices]
            for subset in itertools.combinations(paths, self.n):
                samples.append([subset, t])
        return samples

    def __getitem__(self, index):
        target = self.targets[index]
        if self.train:  # select random images to go with it
            possible_choices = self.targets2indices[target]
            if len(possible_choices) <= self.n:
                paths = [self.image_paths[i] for i in possible_choices]
                unique_requirement = False
            else:
                paths = [self.image_paths[index]]
                unique_requirement = True

            while len(paths) < self.n:
                selection = np.random.choice(possible_choices)
                path = self.image_paths[selection]
                if selection not in paths or not unique_requirement:
                    paths.append(path)
        else:
            paths = self.image_paths[index]
        target = torch.ones((self.n, )).long() * target
        images = torch.stack([self.transform(Image.open(p).convert('RGB')) for p in paths])
        return images, target, paths

    def __len__(self):
        return len(self.samples)