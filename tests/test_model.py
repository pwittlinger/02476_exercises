import pytest
import torch

from src.models.model import MyAwesomeModel
from tests import _PATH_DATA, _PROJECT_ROOT

model = MyAwesomeModel()
model_checkpoint = "trained_model.pt"
model.load_state_dict(torch.load(_PROJECT_ROOT+'/models/'+model_checkpoint))


def test_model():
    test = torch.randn(20,1,28,28)
    y = model(test)
    assert y.shape[1] == 10, "Output shape has wrong length (expected len == 10)"


def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match="Expected input is a 4D tensor."):
        model(torch.randn(1,2,3))