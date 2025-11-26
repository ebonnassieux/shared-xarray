### code based on example kindly provided by S.J. Perkins, SARAO
from multiprocessing.shared_memory import SharedMemory
import ctypes
import os
from typing import Tuple
import weakref
import numpy as np
from uuid import uuid4
from xarray import DataArray

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


class SharedArray():
  """
  docstring here
  """

  def __getitem__(self, index):
    print("Getting vals associated with index",index)
    return self._array[index]
  
  def __setitem__(self,index,newvalue):
    self._array[index]=newvalue

  def __str__(self):
    return self._array.values.__str__()

### figure out the below
#  def __repr__(self):
#    return "SharedXarray("+self._array.values.__str__()+")"

  def __init__(self,
               name_str           : str,
               shape              : Tuple[int, ...],
               dtype              : npt.DTypeLike=float,
               writeable          : bool=True,
               verbose            : bool=False,
               ms_async_flag      : bool=False,
               ms_sync_flag       : bool=False,
               ms_invalidate_flag : bool=False,
               op                 : str = "Create"
               ):
    # set flags
    self.MS_ASYNC=ms_async_flag
    self.MS_SYNC=ms_sync_flag
    self.MS_INVALIDATE=ms_invalidate_flag
    # parse input string to replicate SharedArray behaviour
    if "//" in name_str:
      prefix,name = name_str.split("//")
    else:
      prefix=""
      name=name_str
    if prefix== "file:":
      ### TODO implement file backend
      print("File backend not supported yet")
      return NotImplemented
    else:
      # create if not yet existing
      if op=="Create":
        # define shared memory buffer params
        dt = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dt.itemsize
        self.arr_shape=shape
        try:
          self._shm = SharedMemory(create=True, size=nbytes, name=name)
          finalize_fn = unlink_shm
        except FileExistsError:
          if verbose:
            print("Shared memory file already exists")
          return
        # set array value.
        self._array = DataArray(np.ndarray(shape, dt, self._shm.buf))
        #self._array.setflags(write=writeable)
        if verbose==True:
          print(f"{op} {self} in {os.getpid()}")
      # attach if requested.
      elif op=="Attach":
        # attach if already existing
        try:
          self._shm = SharedMemory(create=False, name=name)
          dt = np.dtype(dtype)
          ##### THE PROBLEM RIGHT NOW IS CORRECTLY INHERITING
          ##### THE SHAPE WHEN ATTACHING. OTHERWISE, IT WORKS. 
#          shape = self._shm.buf.size//dt.itemsize
          print(shape)
          shape=[]
          self._array = DataArray(np.ndarray(shape,dt,self._shm.buf))
          #assert self._shm.buf.nbytes == nbytes
          finalize_fn = close_shm
        except FileNotFoundError:
          if verbose:
            print("Shared memory file does not yet exist")
          return
        else:
          if verbose:
            print("Operation not supported")
            return NotImplemented
#    print(self._shm.buf)
#    stop
    # set array value
    self._array = DataArray(np.ndarray(shape, dt, self._shm.buf))
    # define weakref for clean bookkeeping purposes
    weakref.finalize(self, finalize_fn, self._shm.name)
    self._verbose = verbose
    if verbose==True:
      print(f"{op} {self} in {os.getpid()}")

  # make class creation method to ensure .create() call
  # mirrors reference SharedArray behaviour
  @classmethod
  def create(cls,
             name,
             shape):
    return cls(name, shape, op="Create")

  # make class attachment method to ensure .attach() call
  # mirrors reference SharedArray behaviour
  @classmethod
  def attach(cls,
             name):
    return cls(name,shape=[], op="Attach")

  @property
  def shape(self) -> Tuple[int, ...]:
    return self._array.shape
  
  def delete(self,name):
    """
    docstring here
    """
    return
  

  def list(self):
    """
    docstring here
    """
    return
