import os
import gzip
import hashlib
import requests
import numpy as np 
from pathlib import Path

def fetch(url):
  fname = hashlib.md5(url.encode("utf-8")).hexdigest()
  _dir = Path("/tmp/turing")
  _dir.mkdir(parents=True, exist_ok=True)
  fpath = _dir / fname
  if fpath.exists():
    with open(fpath, "rb") as f: 
      return f.read(), fpath
  else:
    with open(fpath, "wb") as f:
      print(f"Downloading the data from {url}")
      resp = requests.get(url)
      resp.raise_for_status()
      f.write(resp.content)
      return resp.content, fpath

def fetch_mnist():
  def _fetch_mnist(url): 
    data, _ = fetch(url)
    data = np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()
    return data

  X_train = _fetch_mnist("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28))
  Y_train = _fetch_mnist("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = _fetch_mnist("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28))
  Y_test = _fetch_mnist("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
  return X_train, Y_train, X_test, Y_test
