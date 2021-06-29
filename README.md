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

For now, this is what it looks like to use. You have to reference a source file and a 'member' object that'll be initialized `from_pretrained`. 

In the future, we won't need to do this, as we'll save this info in config.json.

```python
from auto_anything import AutoAnything

model = AutoAnything.from_hf_hub('nateraw/dummy', member='Autoencoder')
```

## TODO

I have no idea if this is a good idea or not yet, so I'd like to try a few things to see how they feel.

- Update HF Model Mixin to handle saving source code when calling `save_pretrained`.
This way you won't have to point to member/entrypoint, but instead can just use ref by
inspecting the class and its file in relation to the root of src.
- Add ability to point to code stored on github as opposed to HF hub. i.e. similar to what `torchhub` does, but using model weights from HF hub.
- Decide if we need to differentiate different types of objects for different use cases. I.e. should it be `AutoModel`, `AutoTransform`, etc?