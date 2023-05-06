# Description: Test cheeger metric
from metrices.cheeger_metric import CheegerMetric
from dataloaders.neighbors_match_dataloader import NeighborsMatchDataloader
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

def test_neighbors_match_cheeger():
    # Create dataloader
    dataloader = NeighborsMatchDataloader(depth=3, train_fraction=0.8, batch_size=1024, loader_workers=1)
    dataset = dataloader.get_dataset()
    X_train, X_test, dim0, out_dim, criterion = dataset
    batch = Batch.from_data_list(X_test)
    # Run cheeger metric on X_test
    cheeger_metric = CheegerMetric()
    scores = cheeger_metric(batch)
    # Check that the output has the correct length
    assert len(scores) == 1600
    # Check that the output values are correct
    # Compute the Cheeger score manually for each graph
    for i in range(len(X_test)):
        print(i)
        graph = X_test[i]
        score = 4 * cheeger_metric._compute_lambda_1(graph)
        # Check that the computed scores match the output
        assert scores[i] == pytest.approx(score, rel=1e-3)
