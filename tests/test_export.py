"""The exported programs must compute the same thing as the PyTorch model."""

import pytest
import torch

from edgeml.export import export_program
from edgeml.runtime import PteModel

from .test_model import small_model

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model_and_inputs():
    m = small_model()
    ids = torch.randint(4, 100, (1, 16))
    mask = torch.ones_like(ids)
    mask[:, 12:] = 0
    return m, (ids, mask)


def test_fp32_program_matches_pytorch(tmp_path, model_and_inputs):
    m, inputs = model_and_inputs
    rep = export_program(m, inputs, tmp_path / "m_fp32.pte", "fp32")
    assert rep["pte_bytes"] > 0 and rep["num_delegated_subgraphs"] >= 1
    with torch.no_grad():
        torch.testing.assert_close(PteModel(tmp_path / "m_fp32.pte")(*inputs), m(*inputs), atol=1e-4, rtol=1e-4)


def test_int8_program_is_close_and_smaller(tmp_path, model_and_inputs):
    m, inputs = model_and_inputs
    fp = export_program(m, inputs, tmp_path / "m_fp32.pte", "fp32")
    q = export_program(m, inputs, tmp_path / "m_int8.pte", "int8")
    assert q["pte_bytes"] < fp["pte_bytes"]
    with torch.no_grad():
        ref = m(*inputs)
    out = PteModel(tmp_path / "m_int8.pte")(*inputs)
    assert torch.nn.functional.cosine_similarity(out, ref).item() > 0.99
