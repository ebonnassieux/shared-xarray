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
