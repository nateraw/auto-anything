# auto-anything

Playing with ideas to include/reference code on Huggingface's hub. Experimental!


## Setup

Clone the repo and install locally. Still in dev, so I suggest you just install in development mode.

```
git clone https://github.com/nateraw/auto-anything.git
cd auto-anything
pip install -e .
```

## Usage

Add `auto_anything.ModelHubMixin` to your torch module.

```
class Autoencoder(nn.Module, ModelHubMixin):

    def __init__(self, input_dim: int = 784, hidden_dims: Tuple[int] = (256, 64, 16, 4, 2)):
        super().__init__()
        self.config = Namespace(input_dim=input_dim, hidden_dims=hidden_dims)
        self.encoder = Encoder(self.config.input_dim, *self.config.hidden_dims)
        self.decoder = Decoder(self.config.input_dim, *reversed(self.config.hidden_dims))

    def forward(self, x):
        x = x.flatten(1)
        latent = self.encoder(x)
        recon = self.decoder(latent)
        loss = F.mse_loss(recon, x)
        return recon, latent, loss
```

Add `src_dir` kwarg pointing to root of your source code when calling `model.save_pretrained`. This will add it to your Huggingface Hub repo so it can be referenced later. You can also add `requirements`. If a user doesn't have the specified requirements when they go to try and initialize your object later, they'll recieve an error.

```
model.save_pretrained(
    'cool_model',
    src_dir="./src",
    requirements=['torch', 'torchvision', 'auto_anything']
)
```

Finally, initialize from hub. Your source code is included in the cached download from HF hub, so you're always using the correct snapshot of the code.

```python
from auto_anything import AutoAnything

model = AutoAnything.from_pretrained('nateraw/autoencoder-cifar10')
```

## TODO

I have no idea if this is a good idea or not yet, so I'd like to try a few things to see how they feel.

-  [x] Update HF Model Mixin to handle saving source code when calling `save_pretrained`.
This way you won't have to point to member/entrypoint, but instead can just use ref by
inspecting the class and its file in relation to the root of src.
- [ ] Add ability to point to code stored on github as opposed to HF hub. i.e. similar to what `torchhub` does, but using model weights from HF hub.
- [ ] Decide if we need to differentiate different types of objects for different use cases. I.e. should it be `AutoModel`, `AutoTransform`, etc?
- [x] Methods `from_hf_hub` and `from_github` likely will be confusing, and should probably be replaced with the usual `from_pretrained`. The config.json file on HF hub will tell us where to find source code if its available.
