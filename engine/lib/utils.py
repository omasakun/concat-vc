import os
import zipfile
from contextlib import contextmanager
from os import path
from pathlib import Path
from typing import Any, Callable, TypeVar, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import device as TorchDevice
from torch.utils.data import Dataset
from tqdm import tqdm

T = TypeVar("T")

NPArray = NDArray[Any]
Device = Union["TorchDevice", str]

DATA_DIR = Path(__file__).parent.parent.parent / "data"

def extract_zip(src: Path | str, dest: Path | str) -> list[str]:
  with zipfile.ZipFile(src, "r") as zf:
    for file in tqdm(zf.namelist(), ncols=0):
      zf.extract(file, dest)

def make_parents(file: str | Path):
  os.makedirs(path.dirname(file), exist_ok=True)

class FilteredDataset(Dataset[T]):
  def __init__(self, base: Dataset[T], *, fn: Callable[[T], bool] = None, indices: list[int] = None) -> None:
    self._base = base
    self._fn = fn
    self._indices = indices or [i for i, x in tqdm(enumerate(base), total=len(base), desc="FilteredDataset", ncols=0) if fn(x)]

  def __getitem__(self, n: int) -> T:
    return self._base[self._indices[n]]

  def __len__(self) -> int:
    return len(self._indices)

@contextmanager
def timer(desc: str = "Duration"):
  from time import monotonic_ns
  start = monotonic_ns()
  yield
  end = monotonic_ns()
  print(f"{desc}: {(end - start) / 1e6:.3f} ms")

@contextmanager
def hide_warns():
  import logging
  import warnings

  with change_loglevel("transformers", logging.ERROR):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      yield

@contextmanager
def change_loglevel(logger: str, level: int):
  import logging
  prev_level = logging.getLogger(logger).level
  logging.getLogger(logger).setLevel(level)
  yield
  logging.getLogger(logger).setLevel(prev_level)

def np_safesave(file: str | Path, arr: NPArray):
  # First, save to a temporary file
  # Then, rename it to the target file
  # This is to prevent the target file from being corrupted

  np.save(str(file) + ".tmp.npy", arr)
  os.replace(str(file) + ".tmp.npy", file)

def clamp(x: float, mn: float, mx: float) -> float:
  return max(min(x, mx), mn)

def save_ckpt(path: Path | str, data: dict = {}, **kwargs):
  path = str(path)
  make_parents(path)

  ckpt = {"data": data}
  for k, v in kwargs.items():
    ckpt[k] = v.state_dict()

  torch.save(ckpt, path + ".tmp")
  os.replace(path + ".tmp", path)

def load_ckpt(path: Path | str, **kwargs) -> dict:
  path = str(path)
  ckpt = torch.load(path)

  for k, v in kwargs.items():
    v.load_state_dict(ckpt[k])

  return ckpt["data"]
