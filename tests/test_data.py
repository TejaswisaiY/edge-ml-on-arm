import pytest

from edgeml.data import OOS, load_clinc150

pytestmark = pytest.mark.network


def test_clinc150_splits_and_labels():
    d = load_clinc150()
    # 150 intents x (100 train, 20 val, 30 test) + official out-of-scope splits (100 / 100 / 1000).
    assert (len(d.train), len(d.val), len(d.test)) == (15100, 3100, 5500)
    assert d.num_labels == 151
    assert d.label_names[d.oos_id] == OOS == d.label_names[-1]
    assert sum(y == d.oos_id for y in d.test.labels) == 1000
