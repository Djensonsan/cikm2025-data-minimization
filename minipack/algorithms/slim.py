from recpack.algorithms.slim import SLIM as RecPackSLIM
from minipack.algorithms.util.serializable import SerializableModel

class SLIM(SerializableModel, RecPackSLIM):
    """
    A simple wrapper around a RecPack SLIM model with save and load functionality.
    """
    pass