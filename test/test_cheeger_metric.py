# Description: Test cheeger metric
from metrices.cheeger_metric import CheegerMetric
import numpy as np
import pytest
from torch_geometric.data import Data, Batch
import torch

def test_CheegerMetric_call():
    # Create two simple graphs
    edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr1 = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    graph1 = Data(edge_index=edge_index1, edge_attr=edge_attr1, num_nodes=3)

    edge_index2 = torch.tensor([[0, 1, 2], [1, 0, 1]], dtype=torch.long)
    edge_attr2 = torch.tensor([1, 2, 3], dtype=torch.float)
    graph2 = Data(edge_index=edge_index2, edge_attr=edge_attr2, num_nodes=3)

    # Create a batch of the two graphs
    batch = Batch.from_data_list([graph1, graph2])

    # Compute the Cheeger score using the CheegerMetric class
    cheeger_metric = CheegerMetric()
    scores = cheeger_metric(batch)

    # Check that the output has the correct length
    assert len(scores) == 2

    # Check that the output values are correct
    # Compute the Cheeger score manually for each graph
    score1 = 4 * cheeger_metric._compute_lambda_1(graph1)
    score2 = 4 * cheeger_metric._compute_lambda_1(graph2)
    # Check that the computed scores match the output
    assert scores[0] == pytest.approx(score1)
    assert scores[1] == pytest.approx(score2)