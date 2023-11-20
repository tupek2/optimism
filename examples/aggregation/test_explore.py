import jax
import chex
import jax.numpy as np

# data class
from chex._src.dataclass import dataclass
from chex._src import pytypes


@dataclass(frozen=True, mappable_dataclass=True)
class Polyhedral:
    a : pytypes.ArrayDevice
    b: pytypes.ArrayDevice

p = Polyhedral(a=1.0, b=0.2)