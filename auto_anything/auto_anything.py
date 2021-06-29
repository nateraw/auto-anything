import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from .utils import check_requirements


class AutoAnything:

    def __init__(self):
        raise RuntimeError("Don't use __init__ of AutoAnything. Use from_hf_hub or from_github instead")

    @classmethod
    def from_hf_hub(cls, namespace, entrypoint='model.py', member='MessageModel', **kwargs):
        """Initialize pretrained model object using weights + code from the hub.

        TODO - Update HF Model Mixin to handle saving source code when calling save_pretrained().
        This way you won't have to point to member/entrypoint, but instead can just use ref by
        inspecting the class and its file in relation to the root of src.

        Args:
            namespace (str): HF model repo reference namespace. Example: 'nateraw/dummy'.
            entrypoint (str, optional): Entrypoint file we'll import as a module. Defaults to 'model.py'.
            member (str, optional): Name of the object member of the entrypoint module. Defaults to 'MessageModel'.

        Returns:
            Any: Object initialized from_pretrained. Often a model, but can be basically any object.
        """
        cached_path = Path(snapshot_download(namespace))
        requirements_path = Path(cached_path) / 'requirements.txt'
        check_requirements(requirements_path)

        assert (cached_path / entrypoint).exists()
        sys.path.append(str(cached_path))

        # TODO - this won't work if entrypoint isn't at top level of hub repo
        module = __import__(Path(entrypoint).stem)

        # My thought here is what if you add multiple to path...the __import__ would get all wonky.
        sys.path.remove(str(cached_path))


        # What to do here???

        # return module
        return getattr(module, member).from_pretrained(str(cached_path), **kwargs)

    def from_github(self, namespace):
        '''TODO'''
        raise NotImplementedError()
