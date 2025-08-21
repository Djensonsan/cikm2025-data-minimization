import torch
from minipack.algorithms.util.serializable import SerializableModel, SerializableTorchModel

class MockModel(SerializableModel):
    def __init__(self, name, value):
        self.name = name
        self.value = value

class MockTorchModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 1)

    def forward(self, x):
        return self.linear(x)

class MockTorchModel(SerializableTorchModel):
    @property
    def name(self):
        return self.__class__.__name__

    def __init__(self):
        self.model_ = MockTorchModule()