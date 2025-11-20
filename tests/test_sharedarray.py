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
    from shared_xarray.shared_xarray import close_shm, unlink_shm
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
    from shared_xarray.shared_xarray import	unlink_shm
    shm_a = SharedMemory(name="test",create=True, size=10)
    unlink_shm("test")
    # verify file not found error
    try:
        unlink_shm("test")
    except FileNotFoundError:
        return
    except Exception:
        return

        
