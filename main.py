from dataloaders.neighbors_match_dataloader import NeighborsMatchDataloader

dataloader = NeighborsMatchDataloader(depth=3, train_fraction=0.8, batch_size=1024, loader_workers=1)
dataset = dataloader.get_dataset()
X_train, X_test, dim0, out_dim, criterion = dataset
graph = X_test[0]
print()