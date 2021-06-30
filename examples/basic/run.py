from pathlib import Path

import pytorch_lightning as pl
import torch
from huggingface_hub import HfApi, HfFolder, Repository
from src.model import Autoencoder
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class LitAutoencoder(pl.LightningModule):
    def __init__(self, model, lr: float = 1e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters("lr", *kwargs)
        self.model = model
        self.forward = self.model.forward

    def shared_step(self, batch, mode="train"):
        x, _ = batch
        recon, latent, loss = self(x)
        self.log("{mode}_loss", loss)
        return recon, latent, loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)[-1]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def fit(self, *data, **trainer_kwargs):
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self, *data)


if __name__ == "__main__":
    ds = CIFAR10("./", download=True, transform=ToTensor())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = Autoencoder(input_dim=3072)
    estimator = LitAutoencoder(model)
    estimator.fit(loader, gpus=1, limit_train_batches=100, max_epochs=1)

    # Authenticate w/ token + create/clone the repo if you haven't already
    token = HfFolder.get_token()
    repo_url = HfApi().create_repo(token=token, name="autoencoder-cifar10", exist_ok=True)
    model_repo = Repository(
        "./saved_model",
        clone_from=repo_url,
        use_auth_token=token,
        git_email="nateraw",
        git_user="naterawdata@gmail.com",
    )

    # Save model weights and code to local repo
    model.save_pretrained(
        model_repo.local_dir,
        src_dir="./src",
        requirements=[x for x in Path('./requirements.txt').read_text().split('\n') if x != 'pytorch-lightning']
    )

    # Push!
    commit_url = model_repo.push_to_hub()
