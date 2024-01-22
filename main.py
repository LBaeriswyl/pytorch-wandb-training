import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import wandb
import yaml

use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available()

if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def get_dataloaders(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # download MNIST training dataset
    trainset = datasets.MNIST("./cache", train=True, download=True,
                             transform=transform)
    # download MNIST test dataset
    testset = datasets.MNIST("./cache", train=False, download=True,
                             transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


def get_model(fc_layer_size, dropout):
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, fc_layer_size),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(fc_layer_size, 10)
    )
    return model


def get_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    elif optimizer == "adamW":
        optimizer = optim.AdamW(network.parameters(),
                               lr=learning_rate)
    return optimizer


def train_epoch(model, loader, optimizer): 
    model.train()

    total_loss = 0
    total_correct = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        #Forward pass
        pred = model(data)
        loss = F.cross_entropy(pred, target)
        total_loss += loss.item()
        total_correct += pred.argmax(dim=1).eq(target).sum().item()

        #Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    epoch_loss = total_loss / len(loader)
    epoch_acc = total_correct / len(loader.dataset)

    return epoch_loss, epoch_acc


def test_epoch(model, loader):
    model.eval()

    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            #Forward pass
            pred = model(data)
            loss = F.cross_entropy(pred, target)
            total_loss += loss.item()
            total_correct += pred.argmax(dim=1).eq(target).sum().item()
        
    test_loss = total_loss / len(loader)
    test_acc = total_correct / len(loader.dataset)

    return test_loss, test_acc


def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config

        trainloader, testloader = get_dataloaders(config.batch_size)
        model = get_model(config.fc_layer_size, config.dropout)
        optimizer = get_optimizer(model, config.optimizer, config.learning_rate)

        model.to(device)

        for epoch in range(1, config.epochs+1):
            train_epoch_loss, train_epoch_acc = train_epoch(model, trainloader, optimizer)
            test_loss, test_acc = test_epoch(model, testloader)
            wandb.log({"train_loss": train_epoch_loss, "train_acc": train_epoch_acc,
                       "test_loss": test_loss, "test_acc": test_acc, "epoch": epoch})


def wandb_sweep(config, project, count):
    sweep_id = wandb.sweep(config, project=project)
    wandb.agent(sweep_id, train_model, count=count)
    wandb.finish()


def load_config(config_file):
    with open(config_file) as file:
        config=yaml.safe_load(file)
    return config


def main(config_file, dry_run, project, count):
    print(f"Running on device: {device}")
    if dry_run:
        print("Running in dry run mode.")
        os.environ["WANDB_MODE"] = "offline"
    config = load_config(config_file)
    print("Sweep configuration:")
    print(config)
    print("Starting sweep...")
    wandb_sweep(config, project, count)
    print("Sweep completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run locally without logging to wandb")
    parser.add_argument("--project", type=str, help="Wandb project name")
    parser.add_argument("--count", type=int, help="Number of runs to execute")
    args = parser.parse_args()
    main(args.config_file, args.dry_run, args.project, args.count)