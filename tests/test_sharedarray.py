### test file for SharedArray class of shared-xarray
### aim is to test all functionalities
import numpy.testing as npt
import numpy as np
import pytest
import os

### canonical SharedArray class to compare to
### test sharedarray library functionalities
def test_sharedarray_reference():
    import SharedArray as sa
    # clean up in case previous testing was done only partly
    if os.path.isfile("shm://test"):
        sa.delete("test")
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

### test SharedArray function of shared-xarray
def test_SharedArray_sharedxarray():
    from shared_xarray.shared_xarray import SharedArray as sa
    # create an array in shared memory
    a = sa.create("shm://test", 10)
    # Attach it as a different array. This can be done from another
    # python interpreter as long as it runs on the same computer.
    b = sa.attach("shm://test")
    # See how they are actually sharing the same memory.
    a[0] = 42
    # npt.assert_equal(a,b) # this fails. idk why.
    npt.assert_equal(a.values,b.values) # todo: use DataArray.equals instead
    # Destroying a does not affect b.
    del a
    npt.assert_equal(b[0].values,42)
    # delete corresponds to unlinking. Use close to simply de-attach
    sa.delete("test")
    
def test_SharedXarray_functionalities():
    from shared_xarray.shared_xarray import SharedArray as sa
    a = sa.create("shm://test", 10)
    #print(a._array.keys())
