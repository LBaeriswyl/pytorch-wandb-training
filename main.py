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


def get_dataloader(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # download MNIST training dataset
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)
    return loader


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


def train_epoch(network, loader, optimizer):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        #Forward pass
        loss = F.cross_entropy(network(data), target)
        cumu_loss += loss.item()

        #Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)


def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config

        loader = get_dataloader(config.batch_size)
        model = get_model(config.fc_layer_size, config.dropout)
        optimizer = get_optimizer(model, config.optimizer, config.learning_rate)

        model.to(device)
        model.train()

        for epoch in range(1, config.epochs+1):
            avg_loss = train_epoch(model, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})


def wandb_sweep(config, dry_run, project):
    sweep_id = wandb.sweep(config, project=project)
    wandb.agent(sweep_id, train_model)
    wandb.finish()


def load_config(config_file):
    with open(config_file) as file:
        config=yaml.safe_load(file)
    return config


def main(config_file, dry_run, project):
    print(f"Running on device: {device}")
    if dry_run:
        print("Running in dry run mode.")
        os.environ["WANDB_MODE"] = "offline"
    config = load_config(config_file)
    print("Sweep configuration:")
    print(config)
    print("Starting sweep...")
    wandb_sweep(config, dry_run, project)
    print("Sweep completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run locally without logging to wandb")
    parser.add_argument("--project", type=str, help="Wandb project name")
    args = parser.parse_args()
    main(args.config_file, args.dry_run, args.project)