from recpack.algorithms.nearest_neighbour import ItemKNN as RecPackItemKNN
from recpack.algorithms.nearest_neighbour import ItemPNN as RecpackItemPNN

from minipack.algorithms.util.serializable import SerializableModel

class ItemKNN(SerializableModel, RecPackItemKNN):
    """
    A simple wrapper around a RecPack ItemKNN model with save and load functionality.
    """
    pass


class ItemPNN(SerializableModel, RecpackItemPNN):
    """
    A simple wrapper around a RecPack ItemPNN model with save and load functionality.
    """
    pass