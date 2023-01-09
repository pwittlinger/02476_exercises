from tests import _PATH_DATA
import pytest
import os, sys
#script_path = os.path.realpath(os.path.dirname(__name__))
#os.chdir(script_path)
#sys.path.append("..")
from src.data.data import CorruptMnist
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    dataset_train = CorruptMnist(train=True)
    dataset_test = CorruptMnist(train=False)
    N_train = 25000
    N_test = 5000
    assert len(dataset_train) == N_train, "Train data does not have correct number of entries. Expected 25k"
    assert len(dataset_test) == N_test, "Test data does not have correct number of entries."
    assert dataset_train.data.size(dim=1) == 1
    assert dataset_train.data.size(dim=2) == 28
    assert dataset_train.data.size(dim=3) == 28
    assert [dataset_train.data.size(dim=1),dataset_train.data.size(dim=2),dataset_train.data.size(dim=3)] == [1,28,28], "Input shape of train data is wrong."
    assert [dataset_test.data.size(dim=1),dataset_test.data.size(dim=2),dataset_test.data.size(dim=3)] == [1,28,28], "Input shape of test data is wrong."
    assert dataset_test.data.size(dim=1) == 1
    assert dataset_test.data.size(dim=2) == 28
    assert dataset_test.data.size(dim=3) == 28
#assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
#assert that all labels are represented
