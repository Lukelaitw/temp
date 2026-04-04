from importlib.metadata import PackageNotFoundError, version

from .vae import ScviVAE, TransformerVAE

__all__ = [
    "ScviVAE",
    "TransformerVAE",
]

try:
    __version__ = version("scldm")
except PackageNotFoundError:
    __version__ = "0.0.0"
