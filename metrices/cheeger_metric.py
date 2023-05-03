class CheegerMetric():
    def __init__(self, name='cheegr'):
        super().__init__(name)

    # def _compute(self, y_true, y_pred):
    #     return cheeger_score(y_true, y_pred)
    
    def _compute_edge_boundary(self, sub_graph, graph_edges):
        # :param sub_graph: list of vertices
        # :param graph_edges: list of edges
        # :return: edge boundary
        sub_graph_edges = []
        edge_boundary = 0
        for edge in graph_edges:
            if edge[0] in sub_graph and edge[1] not in sub_graph or edge[0] not in sub_graph and edge[1] in sub_graph:
                edge_boundary += 1
        return edge_boundary
    
    def get_graph_value(self, graph):
        # Input is a torch_geometric.data.Data object
        # Compute the metric for a single graph
        # Compute the edge boundary for the graph
        edges = graph.edge_index.t()
        vertices = graph.x.shape[0]
        # Compute powerset of vertices
        relevant_sets = self.get_relevant_sets(vertices)
        # Compute the edge boundary for each subgraph
        edge_boundary = []
        for sub_graph in relevant_sets:
            edge_boundary.append(self._compute_edge_boundary(graph, sub_graph, edges))
        # Compute the cheeger constant
        cheeger = []
        for i in range(len(edge_boundary)):
            cheeger.append(edge_boundary[i] / len(sub_graph))
        # Return the minimum cheeger constant
        return min(cheeger)

    def get_relevant_sets(self, iterable):
        # powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        relevant_sets = []
        # Set size up to 0.5 * |V|
        for i in range(int(0.5 * len(iterable))): 
            relevant_sets.append(list(self.combinations(range(iterable), i)))
        return relevant_sets

    def combinations(self, iterable, r):
        # combinations('ABCD', 2) --> AB AC AD BC BD CD
        # combinations(range(4), 3) --> 012 013 023 123
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = list(range(r))
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1
            yield tuple(pool[i] for i in indices)

    def get_average_value(self, X_test):
        # Iterate over all graphs
        cheegers = []
        for i in range(len(X_test)):
            cheegers.append(self._compute(X_test[i].y, X_test[i].pred))