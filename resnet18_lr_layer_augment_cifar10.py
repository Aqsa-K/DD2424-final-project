from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from gcs_functions import *
from google.cloud import storage
import json


client = storage.Client()
write_to_storage('resnet_18_experiment_test_11_5', 'beginning_log.txt', 'Beginning of the experiment')


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train' and scheduler is not None:
                if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Store metrics
            if phase == 'train':
                metrics['train_loss'].append(epoch_loss)
                metrics['train_acc'].append(epoch_acc.item())
            if phase == 'val':
                metrics['val_loss'].append(epoch_loss)
                metrics['val_acc'].append(epoch_acc.item())

            # Deep copy the model if best accuracy is achieved
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics


def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    model_ft = None
    input_size = 0

    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    model_ft = model_ft.to(device)

    return model_ft, input_size


"""#### Learning Rate Experiments"""


def create_optimizer(model, lr_main, lr_fc):
    params_to_update = []
    param_groups = [
        {"params": [], "lr": lr_main},
        {"params": [], "lr": lr_fc}
    ]

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name:
                param_groups[1]["params"].append(param)
            else:
                param_groups[0]["params"].append(param)

    optimizer = optim.Adam(param_groups)
    return optimizer


def lr_experiments(lrs):
    results = {
        "lr_main": [],
        "lr_fc": [],
        "scheduler_type": [],
        "final_acc": []
    }

    i = 0
    for lr_main, lr_fc, scheduler_type in lrs:
        print('Experiment {}, {}'.format(i + 1, lrs[i]))

        model_ft, input_size = initialize_model(num_classes, feature_extract=False)
        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = create_optimizer(model_ft, lr_main, lr_fc)
        if scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        elif scheduler_type == "exp":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2,
                                                                   threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                                   eps=1e-04)
        elif scheduler_type == "cycle":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=100,
                                                          step_size_down=100, cycle_momentum=False)
        else:
            scheduler = None

        trained_model, metrics = train_model(model_ft, dataloaders_dict, criterion, optimizer, scheduler,
                                             num_epochs)
        results["lr_main"].append(lr_main)
        results["lr_fc"].append(lr_fc)
        results["scheduler_type"].append(scheduler_type)
        results["final_acc"].append(metrics)
        i += 1

    return results


"""#### Fine Tuning Layers Experiment"""


def get_model(num_classes, layers_to_tune):
    # Load a pretrained model
    model = models.resnet18(pretrained=True)

    # Freeze all layers in the network
    for param in model.parameters():
        param.requires_grad = False

    # Layer groups in ResNet18
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    for layer_index in layers_to_tune:
        for param in getattr(model, layer_names[layer_index - 1]).parameters():
            param.requires_grad = True

    # Replace the final fully connected layer (unfrozen)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # CIFAR10 has 10 classes
    model = model.to(device)

    return model


def get_params_to_update(model):
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    return params_to_update


def run_layer_fine_tuning_experiments():
    ft_results = {
        "layers_to_tune": [],
        "final_acc": []
    }

    experiments = [
        {"layers_to_tune": [4]},
        {"layers_to_tune": [3, 4]},
        {"layers_to_tune": [2, 3, 4]},
        {"layers_to_tune": [1, 2, 3, 4]},
    ]
    i = 1
    for experiment in experiments:
        model = get_model(num_classes, experiment["layers_to_tune"])
        params_to_update = get_params_to_update(model)
        optimizer_ft = torch.optim.Adam(params_to_update, lr=0.0001)
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        print("Experiment {} with layer number {} tuned".format(i, str(experiment["layers_to_tune"])))
        model_ft, metrics = train_model(model, dataloaders_dict, criterion, optimizer_ft, scheduler=None,
                                        num_epochs=num_epochs)
        print(f'Experiment with layers {experiment["layers_to_tune"]} completed.')
        i += 1
        ft_results["layers_to_tune"].append(experiment["layers_to_tune"])
        ft_results["final_acc"].append(metrics)

    return ft_results


"""#### Data Augmentation Experiments"""


def data_augmentation_experiments(testloader, num_classes, batch_size, sub_length, augmentation_types, lr_main, lr_fc):
    results = {
        "augmentation_type": [],
        "final_acc": []
    }

    i = 0
    for augmentation_type in augmentation_types:
        print('Experiment {}: {}'.format(i + 1, augmentation_type))
        model_ft, input_size = initialize_model(num_classes, feature_extract=False)
        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(model_ft, lr_main, lr_fc)
        # scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
        # Update transforms based on augmentation type
        transform = get_transform(augmentation_type)
        full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)
        subset_size = int(sub_length * len(full_dataset))
        subset_indices = torch.randperm(len(full_dataset))[:subset_size]
        subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
        trainloader = torch.utils.data.DataLoader(subset_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        dataloaders_dict = {'train': trainloader, 'val': testloader}

        trained_model, metrics = train_model(model_ft, dataloaders_dict, criterion, optimizer, scheduler=None,
                                             num_epochs=num_epochs)
        results["augmentation_type"].append(augmentation_type)
        results["final_acc"].append(metrics)
        i += 1
    return results


def get_transform(augmentation_type):
    if augmentation_type == "flip":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augmentation_type == "rotation":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augmentation_type == "crops":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augmentation_type == "scaling":
        return transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.RandomCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError("Unsupported augmentation type")


if __name__ == '__main__':
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"

    # Number of classes in the dataset
    num_classes = 10

    # Batch size for training
    batch_size = 512

    # Number of epochs to train for
    num_epochs = 10

    # Percentage of the total dataset
    subset_percentage = 1.0

    # Flag for feature extracting.
    # When False, we finetune the whole model, when True we only update the reshaped layer params.
    feature_extract = False

    mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    # These values are mostly used by researchers as found to very useful in fast convergence
    img_size = 224
    crop_size = 224

    transform = transforms.Compose(
        [
            transforms.Resize(img_size),  # , interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(crop_size),
            # transforms.RandomRotation(20),
            # transforms.RandomHorizontalFlip(0.1),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            # transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
        ])

    transformTest = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
    subset_size = int(subset_percentage * len(full_dataset))
    subset_indices = torch.randperm(len(full_dataset))[:subset_size]
    subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    trainloader = torch.utils.data.DataLoader(subset_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transformTest)
    subset_size_test = int(subset_percentage * len(testset))
    subset_indices = torch.randperm(len(testset))[:subset_size_test]
    subset_testset = torch.utils.data.Subset(testset, subset_indices)
    testloader = torch.utils.data.DataLoader(subset_testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    dataloaders_dict = {'train': trainloader, 'val': testloader}

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Learning Rate Experiments

    learning_rates_schedulers = [
        (0.0001, 0.01, "step"),
        (0.0001, 0.01, "exp"),
        (0.0001, 0.01, "cosine"),
        (0.0001, 0.01, "plateau"),
        (0.0001, 0.01, "cycle"),
        (0.0001, 0.0001, "step"),
        (0.0001, 0.0001, "exp"),
        (0.0001, 0.0001, "cosine"),
        (0.0001, 0.0001, "plateau"),
        (0.0001, 0.0001, "cycle")
    ]

    # Learning Rate Experiments
    results = lr_experiments(learning_rates_schedulers)
    # print(results)
    results_json = json.dumps(results)
    write_json_to_gcs('resnet_18_experiment_test_11_5', 'lr_results.json', results_json)

    # Fine Tuning Layers Experiment
    ft_results = run_layer_fine_tuning_experiments()
    ft_results_json = json.dumps(ft_results)
    write_json_to_gcs('resnet_18_experiment_test_11_5', 'ft_results.json', ft_results_json)

    # Data Augmentation Experiments
    lr_main = 0.0001
    lr_fc = 0.0001
    augmentation_types = ["flip", "rotation", "crops", "scaling"]
    data_augmentation_results = data_augmentation_experiments(testloader, num_classes, batch_size, subset_percentage, augmentation_types, lr_main,
                                                              lr_fc)
    # print(data_augmentation_results)
    data_augmentation_results_json = json.dumps(data_augmentation_results)
    write_json_to_gcs('resnet_18_experiment_test_11_5', 'da_results.json', data_augmentation_results_json)
