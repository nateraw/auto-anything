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

For now, this is what it looks like to use. You have to reference a source file and a 'member' object that'll initialized `from_pretrained`. 

In the future, we won't need to do this, as we'll save this info in config.json.

```python
from auto_anything import AutoAnything

model = AutoAnything.from_hf_hub('nateraw/dummy', member='Autoencoder')
```
