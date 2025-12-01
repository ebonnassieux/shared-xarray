### code based on example kindly provided by S.J. Perkins, SARAO
from multiprocessing.shared_memory import SharedMemory
import ctypes
import os
from typing import Tuple
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
  Function aiming to replace SharedArray
  """
  
  def __getitem__(self, index):    
    return self._array[index]
  
  def __setitem__(self,index,newvalue):
    self._array[index]=newvalue

  def __str__(self):
    return self._array.values.__str__()

  def __repr__(self):
    return f"SharedXarray(name={self.name}, {self._array.values.__str__()})"

  # below, ro means readonly
  def __init__(self,
               name_str           : str,
               shape              : Tuple[int, ...],
               dtype              : npt.DTypeLike=float,
               ro                 : bool=False,
               verbose            : bool=False,
               ms_async_flag      : bool=False,
               ms_sync_flag       : bool=False,
               ms_invalidate_flag : bool=False,
               op                 : str = "Create"
               ):
    ### todo, implement readonly
    if ro==True:
      print("Readonly mode not supported yet")
      return NotImplemented
    # set flags
    self.MS_ASYNC=ms_async_flag
    self.MS_SYNC=ms_sync_flag
    self.MS_INVALIDATE=ms_invalidate_flag
    # parse input string to replicate SharedArray behaviour
    if "//" in name_str:
      self.prefix,self.name = name_str.split("//")
    else:
      self.prefix=""
      self.name=name_str
    if self.prefix== "file:":
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
          self._shm = SharedMemory(create=True, size=nbytes, name=self.name)
        except FileExistsError:
          if verbose:
            print("Shared memory file already exists")
          return
        # set array value.
        self._array = DataArray(np.ndarray(shape, dt, self._shm.buf))
        self._shm.shared_xarray_shape = shape
        self._shm.shared_xarray_dtype = dtype
        #self._array.setflags(write=writeable)
        if verbose==True:
          print(f"{op} {self} in {os.getpid()}")
        # add params to global variable
      # attach if requested.
      elif op=="Attach":
        # attach if already existing
        try:
          self._shm = SharedMemory(create=False, name=self.name)
          array = np.frombuffer(self._shm.buf, dtype)
          shape=array.shape
          dt = np.dtype(dtype)
          self._array = DataArray(np.ndarray(shape,dt,self._shm.buf))
        except FileNotFoundError:
          if verbose:
            print("Shared memory file does not yet exist")
          return
        else:
          if verbose:
            print("Operation not supported")
            return NotImplemented
    # set array value
    self._array = DataArray(np.ndarray(shape, dt, self._shm.buf))
    # define weakref for clean bookkeeping purposes
    self._verbose = verbose
    if verbose==True:
      print(f"{op} {self} in {os.getpid()}")

  # make class creation method to ensure .create() call
  # mirrors reference SharedArray behaviour
  @classmethod
  def create(cls,
             name,
             shape,
             ro=False):
    return cls(name, shape, op="Create",ro=ro)

  # make class attachment method to ensure .attach() call
  # mirrors reference SharedArray behaviour
  @classmethod
  def attach(cls,
             name,
             ro=False):
    # read shape from the input array name
    return cls(name,shape=[], op="Attach",ro=ro)

  @property
  def shape(self) -> Tuple[int, ...]:
    return self._array.shape

  @property
  def values(self) -> array:
    return self._array.values



######## delete functions
  
  
  @classmethod
  def delete(cls,
             name):
    """
    Unlink shared memory
    """
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


  def list(self):
    """
    docstring here
    """
    return NotImplemented
