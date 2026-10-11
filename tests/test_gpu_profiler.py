import pytest
import torch
from torch import nn
from frontier_embedder.gpu_profiler import profile_sequence, profile_sweep, results_as_dicts

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 12)
    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids) * attention_mask.unsqueeze(-1)

@pytest.mark.parametrize('length', [3, 7, 16])
def test_inference(length):
    result = profile_sequence(Toy(), sequence_length=length)
    assert result.status == 'ok'
    assert result.elapsed_seconds >= 0
    assert result.peak_allocated_mb is None

def test_backward():
    model = Toy()
    result = profile_sequence(model, sequence_length=8, mode='backward')
    assert result.status == 'ok'
    assert all(p.grad is None for p in model.parameters())

def test_mode_restored():
    model = Toy()
    model.train()
    profile_sequence(model, sequence_length=4)
    assert model.training
    model.eval()
    profile_sequence(model, sequence_length=4, mode='backward')
    assert not model.training

def test_sweep():
    results = profile_sweep(Toy(), sequence_lengths=(2, 4, 8))
    assert [r.sequence_length for r in results] == [2, 4, 8]
    assert len(results_as_dicts(results)) == 3

@pytest.mark.parametrize('kwargs', [{'sequence_length':0}, {'sequence_length':2,'batch_size':0}, {'sequence_length':2,'vocab_size':1}, {'sequence_length':2,'mode':'invalid'}, {'sequence_length':2,'mode':'backward','warmup':True}])
def test_bad_args(kwargs):
    with pytest.raises(ValueError):
        profile_sequence(Toy(), **kwargs)

def test_frozen_backward_rejected():
    model=Toy()
    for p in model.parameters(): p.requires_grad_(False)
    with pytest.raises(ValueError):
        profile_sequence(model,sequence_length=2,mode='backward')

def test_empty_sweep_rejected():
    with pytest.raises(ValueError):
        profile_sweep(Toy(), sequence_lengths=())
