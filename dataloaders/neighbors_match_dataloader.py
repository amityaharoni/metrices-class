from dataloaders.datasets.dictionary_lookup import DictionaryLookupDataset
from torch_geometric.data import DataLoader

class NeighborsMatchDataloader:
    def __init__(self, depth=3, train_fraction=0.8, batch_size=1024, loader_workers=1):
        self.batch_size = batch_size
        self.loader_workers = loader_workers
        
        print("Creating NeighborsMatchDataset with depth: " + str(depth) + " and train_fraction: " + str(train_fraction))
        self.dataset_class = DictionaryLookupDataset(depth)
        print("Generating data")
        self.dataset = self.dataset_class.generate_data(train_fraction)
        self.X_train, self.X_test, self.dim0, self.out_dim, self.criterion = self.dataset
        print("Creating dataloader")
        self.dataloader = DataLoader(self.X_train * 100, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.loader_workers)


    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataloader)

    def get_dataset(self):
        return self.dataset
    
    def get_dataloader(self):
        return self.dataloader
