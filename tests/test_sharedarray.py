# this replicates the tests in sharedarray dev repo
import numpy.testing as npt
import numpy as np
import SharedArray as sa
from multiprocessing.shared_memory import SharedMemory
import pytest
import os

### test sharedarray library functionalities
import SharedArray as sa
# clean up in case previous testing was done only partly
if os.path.isfile("shm://test"):
    sa.delete("test")
def test_sharedarray():
    import SharedArray as sa
    # Create an array in shared memory.
    a = sa.create("shm://test", 10)
    # Attach it as a different array. This can be done from another
    # python interpreter as long as it runs on the same computer.
    b = sa.attach("shm://test")
    # See how they are actually sharing the same memory.
    a[0] = 42
    npt.assert_array_equal(a,b)
    # Destroying a does not affect b.
    del a
    npt.assert_equal(b[0],42)
    sa.delete("test")

### test global functions

# closer is needed when shared memory is no longer needed by 1 instance
def test_close_shm():
    from shared_xarray.shared_xarray import close_shm
    # create shared memory object to check the closure is done properly
    shm_a = SharedMemory(name="test",create=True, size=10)
    # verify it gets closed properly 
    close_shm("test")
    npt.assert_equal(os.path.isfile("shm://test"),False)
    # verify file not found error
    shm_a.unlink()
    shm_a.close()
    try:
        close_shm("test")
    except FileNotFoundError:
        return

# unlinker is needed when shared memory is no longer needed at all
def test_unlink_shm():
    from shared_xarray.shared_xarray import unlink_shm
    shm_a = SharedMemory(name="test",create=True, size=10)
    unlink_shm("test")
    # verify file not found error
    try:
        unlink_shm("test")
    except FileNotFoundError:
        return
    # verify how to create a test case where the unlinking just fails
    try:
        unlink_shm("test")
    except Exception:
        return

# test shared buffer class
from shared_xarray.shared_xarray import SharedBuffer
def test_sharedbuffer_initialisation():
    shm_a = SharedMemory(name="test_buffer",create=True, size=10)
    test_buffer_1 = SharedBuffer(shm_a)
    npt.assert_equal(shm_a,test_buffer_1._shm)
    test_buffer_2 = SharedBuffer.from_name("test_buffer")
    npt.assert_equal(test_buffer_1.name,test_buffer_2.name)
    shm_a.unlink()
    del(test_buffer_1,test_buffer_2)
def test_sharedbuffer_properties():
    shm_a = SharedMemory(name="test_buffer",create=True, size=10)
    test_buffer_properties = SharedBuffer(shm_a)
    npt.assert_equal(test_buffer_properties.name,"test_buffer")
    del(test_buffer_properties)
    shm_a.unlink()
def test_sharedbuffer_attributes():
    shm_a = SharedMemory(name="test_buffer",create=True, size=10)
    test_buffer_attributes = SharedBuffer(shm_a)
    assert "_shm","__weakref__" in test_buffer_attributes.__slots__
    npt.assert_equal(test_buffer_attributes.__len__(),len(shm_a.buf))
    del(test_buffer_attributes)
    shm_a.unlink()
def test_sharedbuffer_reduce_and_load():
    shm_b = SharedMemory(name="test_buffer_readwrite",create=True, size=10)
    test_buffer_readwrite = SharedBuffer(shm_b)
    # test pickling of sharedbuffer
    test_buffer_readwrite.__reduce__
    # test reconstruction from name
    del(test_buffer_readwrite)
    shm_b.close()
    test_buffer_2 = SharedBuffer.from_name("test_buffer_readwrite")
    npt.assert_equal(test_buffer_2.__len__(),10)
    shm_b.unlink()

### TODO; i dont understand this function.
# test buffer creation from shared array
#def test_from_shared_array():
#    from shared_xarray.shared_xarray import from_shared_array
#    shm = SharedMemory(name="test_buffer_sharedarray",create=True, size=10)
#    test_buffer = SharedBuffer(shm)
#    shape = from_shared_array(shape=10,buffer=test_buffer)


def test_SharedArray():
    
