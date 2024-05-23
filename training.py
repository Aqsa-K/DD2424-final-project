import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split


def get_train_transforms(input_shape, image_size):
    return transforms.Compose([
        transforms.ToTensor(),  # Convert the image back to a tensor
        # transforms.Resize((input_shape[0] + 20, input_shape[0] + 20)),  # Resize to INPUT_SHAPE + 20
        # transforms.RandomCrop((image_size, image_size)),  # Random crop to IMAGE_SIZE
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
    ])


def get_test_transforms(image_size):
    return transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((image_size, image_size)),  # Resize to IMAGE_SIZE
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def prepare_data_cifar(data_dir=None, input_shape=None, image_size=None, batch_size=None):
    train_transform = get_train_transforms(input_shape, image_size)
    test_transform = get_test_transforms(image_size)

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_dataloader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader, train_set, test_set

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split


def get_train_transforms(input_shape, image_size):
    return transforms.Compose([
        transforms.ToTensor(),  # Convert the image back to a tensor
        # transforms.Resize((input_shape[0] + 20, input_shape[0] + 20)),  # Resize to INPUT_SHAPE + 20
        # transforms.RandomCrop((image_size, image_size)),  # Random crop to IMAGE_SIZE
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
    ])


def get_test_transforms(image_size):
    return transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((image_size, image_size)),  # Resize to IMAGE_SIZE
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def prepare_data_cifar(data_dir=None, input_shape=None, image_size=None, batch_size=None):
    train_transform = get_train_transforms(input_shape, image_size)
    test_transform = get_test_transforms(image_size)

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_dataloader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader, train_set, test_set

import torch
import torch.nn as nn
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch.optim.lr_scheduler import LRScheduler
import torch.optim as optim


class WarmUpCosine(LRScheduler):
    def __init__(self, optimizer: optim,
                 total_steps: int,
                 warmup_steps: int,
                 learning_rate_base: float,
                 warmup_learning_rate: float,
                 last_epoch: int = -1):
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = np.pi
        super(WarmUpCosine, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step > self.total_steps:
            return [0.0 for _ in self.base_lrs]

        cos_annealed_lr = 0.5 * self.learning_rate_base * (1 + np.cos(self.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))

        if step < self.warmup_steps:
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            warmup_rate = slope * step + self.warmup_learning_rate
            return [warmup_rate for _ in self.base_lrs]
        else:
            return [cos_annealed_lr for _ in self.base_lrs]


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        # print('img shape', img.shape)
        patches = self.patchify(img)
        # print('patches shape', patches.shape)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes


class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=6,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size // patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T - 1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask


class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=6,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask


class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits

from google.cloud import storage
from PIL import Image

import os
import matplotlib.pyplot as plt
import numpy as np
import random
import tempfile

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split

# from load_data import *
# from model_mae_timm import *
import io

from google.cloud import storage

BUCKET_NAME = 'experiment_results123'


def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a PIL Image."""
    tensor = tensor.cpu().detach()  # Move tensor to CPU and detach from gradients
    # tensor = (tensor + 1) / 2  # Normalize if required, adjust based on your normalization
    tensor = tensor.clamp(0, 1)  # Clamp values to valid image range
    transform = transforms.ToPILImage()
    image = transform(tensor)
    return image


def upload_blob_from_memory(blob_data, destination_blob_name, content_type):
    """Uploads a file to the bucket from memory."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_file(blob_data, content_type=content_type)

    print(f"Data uploaded to {destination_blob_name}.")


def save_history_to_gcs(history_json, destination_blob_name):
    client = storage.Client()
    bucket_name = BUCKET_NAME
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(history_json, content_type='application/json')
    print("History object saved to GCS")


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob('model' + destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def pre_train(experiment_name, mask_ratio=0.75, decoder_depth=6):
    # for now the input only takes mask_ratio and decoder_depth.
    # for more experiments coming remember to also chang the name for model path, history name and so on.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MAE_ViT(decoder_layer=decoder_depth, mask_ratio=mask_ratio)
    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(),
                              lr=LEARNING_RATE * BATCH_SIZE / 256,
                              betas=(0.9, 0.95),
                              weight_decay=WEIGHT_DECAY)

    total_steps = int((len(train_set) / BATCH_SIZE) * EPOCHS)
    warmup_epoch_percentage = 0.15
    warmup_steps = int(total_steps * warmup_epoch_percentage)

    scheduler = WarmUpCosine(optim, total_steps=total_steps, warmup_steps=warmup_steps, learning_rate_base=LEARNING_RATE, warmup_learning_rate=0.0)

    model_path_pre = './model'
    if not os.path.exists(model_path_pre):
        os.makedirs(model_path_pre)
    model_name = f'mae_pretrain_maskratio_{mask_ratio}_dec_depth_{decoder_depth}.pt'
    model_path = os.path.join(model_path_pre, model_name)
    # the model path should be a model folder that contain all the weights from the pretrained model
    # Moving on to do the classification task, we will load the model from this folder, and use Vit_Classfiier to do the classification task.

    image_path = experiment_name + '/images'

    step_count = 0
    optim.zero_grad()

    history = {
        'experiment': experiment_name,
        'loss': []
    }

    for e in range(EPOCHS):
        model.train()
        losses = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / MASK_PROPORTION
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
        scheduler.step()
        avg_loss = sum(losses) / len(losses)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')
        history['loss'].append(avg_loss)

        # model.eval()
        # with torch.no_grad():
        #     val_img = torch.stack([test_set[i][0] for i in range(8)])
        #     val_img = val_img.to(device)
        #     predicted_val_img, mask = model(val_img)
        #     predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
        #     img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
        #     img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
        #     image = tensor_to_image(img)
        #     image_buffer = io.BytesIO()
        #     image.save(image_buffer, format='JPEG')
        #     image_buffer.seek(0)
        #     upload_blob_from_memory(image_buffer, image_path + f'epoch_{e}.jpg', 'image/jpeg')
        #     # this saves the image to the bucket, inside a folder.

        ''' save '''
        torch.save(model, model_path)
        # Save the model locally
        upload_to_gcs(BUCKET_NAME, model_path, model_name)

    history_json = json.dumps(history)
    save_history_to_gcs(history_json, experiment_name)

import os
import matplotlib.pyplot as plt
import numpy as np
import random
import tempfile

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split

# from load_data import *
# from model_mae_timm import *
# from utils import *
import io
import json

from torch.optim.lr_scheduler import LRScheduler

from google.cloud import storage
from PIL import Image

setup_seed()

# These are some default parameters.
# For the encoder/decoder setting refer to model_mae_timm.py, I initialize the pretraining model using MAE_ViT method.
# OBS!!!: Don't forget to look at the utils.py and change the name of the bucket to your own bucket name.


# DATA
BATCH_SIZE = 512
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
DATA_DIR = './data'

# Optimizer parameters
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 0.05

# Pretraining parameters. Epochs here.
EPOCHS = 100

# Augmentation parameters
IMAGE_SIZE = 32
PATCH_SIZE = 2
MASK_PROPORTION = 0.75

# Encoder and Decoder parameters
LAYER_NORM_EPS = 1e-6



if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)



    train_dataloader, test_dataloader, train_set, test_set = prepare_data_cifar(DATA_DIR, INPUT_SHAPE, IMAGE_SIZE, BATCH_SIZE)
    
    # here is where I do the pretraining.
    
    mask_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85]
    decoder_depths = [2, 4, 6, 8]
    
    # for mask_ratio in mask_ratios:
    #     experiment_name = f'pretrain_mask_ratio_{mask_ratio}_decoder_depth_6'
    #     pre_train(experiment_name, mask_ratio)
    #     print(f'Experiment {experiment_name} is done!')
    #     print('-----------------------------------------------')
    
    for decoder_depth in decoder_depths:
        experiment_name = f'pretrain_mask_ratio_0.75_decoder_depth_{decoder_depth}'
        pre_train(experiment_name, 0.75, decoder_depth)
        print(f'Experiment {experiment_name} is done!')
        print('-----------------------------------------------')
