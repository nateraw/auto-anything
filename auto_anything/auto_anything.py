import importlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from .utils import check_requirements


class AutoAnything:

    def __init__(self):
        raise RuntimeError("Don't use __init__ of AutoAnything. Use from_hf_hub or from_github instead")

    @classmethod
    def from_pretrained(cls, namespace):
        cached_path = Path(snapshot_download(namespace))
        requirements_path = Path(cached_path) / 'requirements.txt'
        check_requirements(requirements_path)

        config_path = cached_path / 'config.json'
        assert config_path.exists()

        with config_path.open(encoding='utf-8') as f:
            config = json.load(f)

        if '_src' in config:
            _src = config.pop('_src')
            module_name = _src['module_name']
            member_name = _src['member_name']

        sys.path.append(str(cached_path / 'hf_src'))
        module = importlib.import_module(module_name)
        sys.path.remove(str(cached_path / 'hf_src'))
        member = getattr(module, member_name)

        return member.from_pretrained(str(cached_path))
