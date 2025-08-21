from recpack.algorithms.factorization_item_similarity import NMFItemToItem as RecPackNMFItemToItem
from recpack.algorithms.factorization_item_similarity import SVDItemToItem as RecPackSVDItemToItem
from minipack.algorithms.util.serializable import SerializableModel

class NMFItemToItem(SerializableModel, RecPackNMFItemToItem):
    """
    A simple wrapper around a RecPack NMF model with save and load functionality.
    """
    pass


class SVDItemToItem(SerializableModel, RecPackSVDItemToItem):
    """
    A simple wrapper around a RecPack SVD model with save and load functionality.
    """
    pass