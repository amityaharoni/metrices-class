from dataloaders.neighbors_match_dataloader import NeighborsMatchDataloader
import torch

# Pytest neighbors match dataloader
def test_neighbors_match_dataloader():
    dataloader = NeighborsMatchDataloader(depth=3, train_fraction=0.8, batch_size=1024, loader_workers=1)
    dataset = dataloader.get_dataset()
    X_train, X_test, dim0, out_dim, criterion = dataset
    assert len(X_train) == 6400
    assert len(X_test) == 1600
    assert dim0 == 8
    assert out_dim == 8
    assert criterion == torch.nn.functional.cross_entropy
    dataloader = dataloader.get_dataloader()
    assert len(dataloader) == 625
    for batch in dataloader:
        assert batch.x.shape == torch.Size([15360, 2])
        assert batch.edge_index.shape == torch.Size([2, 29696])
        assert batch.root_mask.shape == torch.Size([15360])
        assert batch.y.shape == torch.Size([1024])
        break