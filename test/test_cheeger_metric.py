# Test cheeger metric
#
import torch
from torch_geometric.data import Data
from metrices.cheeger_metric import CheegerMetric

edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                            [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
x = torch.tensor([[0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1], [0, 0]], dtype=torch.float)
test_graph = Data(x=x, edge_index=edge_index)

def test_get_relevant_sets():
    # Compute cheeger metric
    cheeger_metric = CheegerMetric()
    relevant_sets = cheeger_metric.get_relevant_sets(test_graph.x.shape[0])
    assert len(relevant_sets) == 4
    assert len(relevant_sets[0]) == 1
    assert len(relevant_sets[1]) == 8
    assert len(relevant_sets[2]) == 28
    assert len(relevant_sets[3]) == 56

def test_combinations():
    # Compute cheeger metric
    cheeger_metric = CheegerMetric()
    combinations = cheeger_metric.combinations(range(test_graph.x.shape[0]), 2)
    assert len(list(combinations)) == 28

def test_compute_edge_boundary():
    # Compute cheeger metric
    cheeger_metric = CheegerMetric()
    edge_boundary = cheeger_metric._compute_edge_boundary(test_graph, test_graph.edge_index.t())
    assert edge_boundary == 8

def test_cheeger_metric():
    # Compute cheeger metric
    cheeger_metric = CheegerMetric()
    cheeger = cheeger_metric.get_graph_value(test_graph)
    assert cheeger == 1.0
