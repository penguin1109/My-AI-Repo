import torch, cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def train_transform(**args):
    aug = transforms.Compose(
        transforms.ToPILImage,
        transforms.Resize(args['img_size']),
        transforms.ToTensor(),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness = (0.7,1.0)),
            transforms.RandomAutoContrast(p= 0.3)
        ], p = 0.5),
        transforms.RandomChoice([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90)
        ], p = 0.5),
        transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
    )
    return aug

def test_transform(**args):
    aug = transforms.Compose(
        transforms.ToPILImage,
        transforms.Resize(args['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    return args


class SiameseDataset(Dataset):
    def __init__(self, dirs, labels = None, **kwargs):
        self.dirs = dirs # image dirs
        self.labels = labels
        self.mode = kwargs['mode']
        self.args = kwargs
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        if self.labels:
            label = self.labels[idx]

        img_dir = self.dirs[idx]
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            transform = train_transform(self.args)
            img = transform(img)
            return{
                'image' : torch.tensor(img, dtype = torch.float32),
                'label' : torch.tensor(self.label_encoder[label], dtype = torch.long)
            }
        else:
            transform = test_transform(self.args)
            img = transform(img)
            return {
                'image' : torch.tensor(img, dtype = torch.float32)
            }
