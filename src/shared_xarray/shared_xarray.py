### code based on example kindly provided by S.J. Perkins, SARAO
from typing import List, Literal
import numpy as np
from xarray_dataclasses import AsDataset, Attr, Coord, Coordof, Data
from multiprocessing.shared_memory import SharedMemory


import ctypes
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process
import os
import pickle
from typing import Tuple
from uuid import uuid4
import weakref

import numpy as np
import xarray


# define shared memory closer function
def close_shm(name: str):
  """Cleanup shared memory"""
  try:
    shm = SharedMemory(name)
  except FileNotFoundError:
    return
  else:
    shm.close()

# define shared memory unlinker
def unlink_shm(name: str):
  """Unlink shared memory"""
  try:
    shm = SharedMemory(name)
  except FileNotFoundError:
    return
  try:
    shm.unlink()
  except Exception:
    print(f"Unlinking {name} failed")
  finally:
    shm.close()

# define functionalities for the shared buffer
class SharedBuffer:
  """ Implements the buffer protocol for a wrapped shared memory buffer
  Pickleable
  """
  __slots__ = ("_shm", "__weakref__")

  _shm: SharedMemory

  @staticmethod
  def from_name(name: str):
    print(f"Reconstructing from {name}")
    return SharedBuffer(SharedMemory(name=name))

  @property
  def name(self) -> str:
    return self._shm.name

  def __reduce__(self):
    print(f"Pickling {self._shm.name}")
    return (SharedBuffer.from_name, (self.name,))

  def __init__(self, shm: SharedMemory):
    self._shm = shm

  def __len__(self):
    return len(self._shm.buf)

  def __buffer__(self, flags):
    return self._shm.buf.__buffer__(flags)

# what this do?
def from_shared_array(shape, dtype, buffer):
  array = np.frombuffer(buffer, dtype)
  array.shape = shape
  return shape

class SharedArray:
  """ Wraps a numpy array backed by shared memory"""
  def __init__(
      self,
      shape: Tuple[int, ...],
      dtype: npt.DTypeLike,
      name: str | None = None,
      writeable: bool = False
  ):
    name_str = name if name else f"array-{uuid4().hex}"
    dt = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dt.itemsize

    # Attach or create
    try:
      self._shm = SharedMemory(name=name_str)
      op = "Attached to"
      assert self._shm.buf.nbytes == nbytes
      finalize_fn = close_shm
    except FileNotFoundError:
      self._shm = SharedMemory(create=True, size=nbytes, name=name_str)
      op = "Created"
      finalize_fn = unlink_shm

    weakref.finalize(self, finalize_fn, self._shm.name)
    self._array = np.ndarray(shape, dt, self._shm.buf)
    if writeable:
      self._array.setflags(write=writeable)

    print(f"{op} {self} in {os.getpid()}")

  def __reduce__(self):
    print(f"Pickling {self.name}")
    return (
      SharedArray, (
        self.shape,
        self.dtype,
        self.name,
        self._array.flags.writeable
      )
    )

  @property
  def name(self):
    return self._shm.name

  @property
  def ndim(self):
    return self._array.ndim

  @property
  def shape(self):
    return self._array.shape

  @property
  def dtype(self):
    return self._array.dtype

  def __array__(self, dtype=None):
    return self._array.astype(dtype=dtype, copy=False) if dtype else self._array

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    xformed_inputs = tuple(i.__array__() if isinstance(i, SharedArray) else i for i in inputs)
    result = self._array.__array_ufunc__(ufunc, method, *xformed_inputs, **kwargs)
    # Need to handle more complex results
    if not isinstance(result, np.ndarray):
      return NotImplemented

    # Copies, not great
    shm_result = SharedArray(result.shape, result.dtype)
    shm_result[:] = result
    return shm_result

  def __array_function__(self, func, types, args, kwargs):
    xformed_types = set(np.ndarray if issubclass(t, SharedArray) else t for t in types)
    xformed_args = tuple(a.__array__() if isinstance(a, SharedArray) else a for a in args)
    result = self._array.__array_function__(func, xformed_types, xformed_args, kwargs)

    # Need to handle more complex results
    if not isinstance(result, np.ndarray):
      return NotImplemented

    # Copies, not great
    shm_result = SharedArray(result.shape, result.dtype)
    shm_result[:] = result
    return shm_result

  def __getitem__(self, key):
    return self._array[key]

  def __setitem__(self, key, value):
    self._array[key] = value

  def __repr__(self):
    buf_ptr = ctypes.c_int.from_buffer(self._array.base).value
    return f"SharedArray(name={self.name}, address={buf_ptr})"

  __str__ = __repr__

if __name__ == "__main__":
  ntime = 100
  na = 7
  nbl = na * (na - 1) // 2
  nchan = 16
  npol = 4

  buf = SharedBuffer(SharedMemory(name=f"buffer-{uuid4().hex}", create=True, size=100))
  buf2: SharedBuffer = pickle.loads(pickle.dumps(buf))
  assert buf._shm.name == buf2._shm.name
  buf._shm.unlink()

  data = xarray.Variable(
    ("time", "baseline", "frequency", "polarization"),
    SharedArray(
      (ntime, nbl, nchan, npol),
      np.complex64,
      writeable=True
    )
  )

  ds = xarray.Dataset({
    "DATA": data,
  },
  coords={
    "time": np.linspace(0.0, 1.0, ntime)
  })

  ds2 = pickle.loads(pickle.dumps(ds))
  ds3 = pickle.loads(pickle.dumps(ds2))
  # Needs SharedArray comparison
  # xarray.testing.assert_identical(ds, ds2)
  # xarray.testing.assert_identical(ds2, ds3)

  ds.DATA[:50] = 10.0

  def proc_fn(dataset):
    assert isinstance(ds.DATA.data, SharedArray)
    print(f"{os.getpid()}: {ds.DATA.data}")
    assert np.all(dataset.DATA.values[:50, ...] == 10.)

  procs = [Process(target=proc_fn, args=(ds,)) for _ in range(5)]

  for p in procs:
    p.start()

  for p in procs:
    p.join()
