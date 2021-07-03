import math
import numpy as np
import matplotlib.pyplot as plt

def calc_dist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

class Node:
    def __init__(self, value):
        self.value = value
        self.weight = None
        self.prev = None
    def __str__(self):
        return str(self.value)

class WeightedGraph:
    def __init__(self):
        self.adj = {}
        self.weights = {}
        self.nodes = {}
    def add_node(self, value):
        self.nodes[value] = Node(value)
        self.adj[value] = []
    def connect(self, value1, value2, weight):
        self.adj[value1].append(value2)
        self.weights[value1, value2] = weight
    def get(self, value):
        return self.nodes[value]

def search_path_with_bellman_ford(weighted_graph, source, target):
    for v in weighted_graph.nodes.values():
        v.prev = None
        v.weight = math.inf
    weighted_graph.get(source).weight = 0
    visited = set()
    visited.add(weighted_graph.get(source))
    for i in range(len(weighted_graph.nodes) - 1):
        for v in weighted_graph.nodes.values():
            if v in visited:
                neighbors = weighted_graph.adj[v.value]
                curr_weight = v.weight
                for neighboring_node in neighbors:
                    node_form = weighted_graph.get(neighboring_node)
                    if curr_weight + weighted_graph.weights[v.value, neighboring_node] < node_form.weight:
                        visited.add(node_form)
                        node_form.weight = curr_weight + weighted_graph.weights[v.value, neighboring_node]
                        node_form.prev = v
    path = []
    source_node_form = weighted_graph.get(source)
    target_node_form = weighted_graph.get(target)
    while target_node_form is not source_node_form:
        path.append(target_node_form)
        target_node_form = target_node_form.prev
    path.append(source)
    path.reverse()
    for i in range(len(path) - 1):
        print(path[i], "->", end=" ")
    print(target)

def add_node_with_split_and_connect(graph, edges, idx, x, y, si, ei, split_length=10):
    split_num = int(calc_dist(x[si], y[si], x[ei], y[ei]) / split_length)
    _sx = np.linspace(x[si], x[ei], split_num)
    _sy = np.linspace(y[si], y[ei], split_num)
    _si = [str(idx[si])+'-'+str(idx[ei])+'_'+str(s) for s in range(split_num)]
    edges.append([_si[0], _si[-1]])
    for s in _si:
        graph.add_node(s)
    for j in range(len(_sx)):
        if j == 0:
            graph.connect(_si[j], _si[j+1], calc_dist(_sx[j], _sy[j], _sx[j+1], _sy[j+1]))
        elif j == len(_sx)-1:
            graph.connect(_si[j], _si[j-1], calc_dist(_sx[j], _sy[j], _sx[j-1], _sy[j-1]))
        else:
            graph.connect(_si[j], _si[j+1], calc_dist(_sx[j], _sy[j], _sx[j+1], _sy[j+1]))
            graph.connect(_si[j], _si[j-1], calc_dist(_sx[j], _sy[j], _sx[j-1], _sy[j-1]))
    return graph, edges

def gen_graph(idx, x, y, split_length=10):
    graph = WeightedGraph()
    edges = []
    for i, _idx in enumerate(idx):
        if i == 0:
            graph, edges = add_node_with_split_and_connect(graph, edges, idx, x, y, i, i+1, split_length)
        elif i == len(idx)-1:
            graph, edges = add_node_with_split_and_connect(graph, edges, idx, x, y, i, i-1, split_length)
        else:
            graph, edges = add_node_with_split_and_connect(graph, edges, idx, x, y, i, i+1, split_length)
            graph, edges = add_node_with_split_and_connect(graph, edges, idx, x, y, i, i-1, split_length)
    for i in range(len(edges)):
        i_start = edges[i][0][0]
        i_end  = edges[i][0][2]
        for j in range(len(edges)):
            if i == j:
                continue
            j_start = edges[j][0][0]
            j_end  = edges[j][0][2]
            if i_start == j_start:
                graph.connect(edges[i][0], edges[j][0], 0)
            if i_start == j_end:
                graph.connect(edges[i][0], edges[j][1], 0)
            if i_end == j_start:
                graph.connect(edges[i][1], edges[j][0], 0)
            if i_end == j_end:
                graph.connect(edges[i][1], edges[j][1], 0)
    return graph

idx = [1, 2, 4, 3, 1]
x = [0, 0, 100, 100, 0]
y = [50, 0, 0, 50, 50]
split_length = 10

graph = gen_graph(idx, x, y, split_length)

search_path_with_bellman_ford(graph, '1-3_1', '4-3_3')

# def initialize_single_source(weighted_graph, source):
#     for v in weighted_graph.nodes.values():
#         v.prev = None
#         v.weight = math.inf
#     weighted_graph.get(source).weight = 0

# def shortest_path_bellman_ford(weighted_graph, source):
#     initialize_single_source(weighted_graph, source)
#     visited = set()
#     visited.add(weighted_graph.get(source))
#     for i in range(len(weighted_graph.nodes) - 1):
#         for v in weighted_graph.nodes.values():
#             if v in visited:
#                 neighbors = weighted_graph.adj[v.value]
#                 curr_weight = v.weight
#                 for neighboring_node in neighbors:
#                     node_form = weighted_graph.get(neighboring_node)
#                     if curr_weight + weighted_graph.weights[v.value, neighboring_node] < node_form.weight:
#                         visited.add(node_form)
#                         node_form.weight = curr_weight + weighted_graph.weights[v.value, neighboring_node]
#                         node_form.prev = v

# def gen_graph(idx, x, y, split_length=10):
#     graph = WeightedGraph()
#     edges = []
#     for i, _idx in enumerate(idx):
#         if i == 0:
#             split_num = int(np.sqrt(calc_dist(x[i], y[i], x[i+1], y[i+1])) / split_length)
#             _sx = np.linspace(x[i], x[i+1], split_num)
#             _sy = np.linspace(y[i], y[i+1], split_num)
#             _si = [str(idx[i])+'-'+str(idx[i+1])+'_'+str(s) for s in range(split_num)]
#             edges.append([_si[0], _si[-1]])
#             for s in _si:
#                 graph.add_node(s)
#             for j in range(len(_sx)):
#                 if j == 0:
#                     graph.connect(_si[j], _si[j+1], calc_dist(_sx[j], _sy[j], _sx[j+1], _sy[j+1]))
#                 elif j == len(_sx)-1:
#                     graph.connect(_si[j], _si[j-1], calc_dist(_sx[j], _sy[j], _sx[j-1], _sy[j-1]))
#                 else:
#                     graph.connect(_si[j], _si[j+1], calc_dist(_sx[j], _sy[j], _sx[j+1], _sy[j+1]))
#                     graph.connect(_si[j], _si[j-1], calc_dist(_sx[j], _sy[j], _sx[j-1], _sy[j-1]))
#         elif i == len(idx)-1:
#             split_num = int(np.sqrt(calc_dist(x[i], y[i], x[i-1], y[i-1])) / split_length)
#             _sx = np.linspace(x[i], x[i-1], split_num)
#             _sy = np.linspace(y[i], y[i-1], split_num)
#             _si = [str(idx[i])+'-'+str(idx[i-1])+'_'+str(s) for s in range(split_num)]
#             edges.append([_si[0], _si[-1]])
#             for s in _si:
#                 graph.add_node(s)
#             for j in range(len(_sx)):
#                 if j == 0:
#                     graph.connect(_si[j], _si[j+1], calc_dist(_sx[j], _sy[j], _sx[j+1], _sy[j+1]))
#                 elif j == len(_sx)-1:
#                     graph.connect(_si[j], _si[j-1], calc_dist(_sx[j], _sy[j], _sx[j-1], _sy[j-1]))
#                 else:
#                     graph.connect(_si[j], _si[j+1], calc_dist(_sx[j], _sy[j], _sx[j+1], _sy[j+1]))
#                     graph.connect(_si[j], _si[j-1], calc_dist(_sx[j], _sy[j], _sx[j-1], _sy[j-1]))
#         else:
#             split_num = int(np.sqrt(calc_dist(x[i], y[i], x[i+1], y[i+1])) / split_length)
#             _sx = np.linspace(x[i], x[i+1], split_num)
#             _sy = np.linspace(y[i], y[i+1], split_num)
#             _si = [str(idx[i])+'-'+str(idx[i+1])+'_'+str(s) for s in range(split_num)]
#             edges.append([_si[0], _si[-1]])
#             for s in _si:
#                 graph.add_node(s)
#             for j in range(len(_sx)):
#                 if j == 0:
#                     graph.connect(_si[j], _si[j+1], calc_dist(_sx[j], _sy[j], _sx[j+1], _sy[j+1]))
#                 elif j == len(_sx)-1:
#                     graph.connect(_si[j], _si[j-1], calc_dist(_sx[j], _sy[j], _sx[j-1], _sy[j-1]))
#                 else:
#                     graph.connect(_si[j], _si[j+1], calc_dist(_sx[j], _sy[j], _sx[j+1], _sy[j+1]))
#                     graph.connect(_si[j], _si[j-1], calc_dist(_sx[j], _sy[j], _sx[j-1], _sy[j-1]))
#             split_num = int(np.sqrt(calc_dist(x[i], y[i], x[i-1], y[i-1])) / split_length)
#             _sx = np.linspace(x[i], x[i-1], split_num)
#             _sy = np.linspace(y[i], y[i-1], split_num)
#             _si = [str(idx[i])+'-'+str(idx[i-1])+'_'+str(s) for s in range(split_num)]
#             edges.append([_si[0], _si[-1]])
#             for s in _si:
#                 graph.add_node(s)
#             for j in range(len(_sx)):
#                 if j == 0:
#                     graph.connect(_si[j], _si[j+1], calc_dist(_sx[j], _sy[j], _sx[j+1], _sy[j+1]))
#                 elif j == len(_sx)-1:
#                     graph.connect(_si[j], _si[j-1], calc_dist(_sx[j], _sy[j], _sx[j-1], _sy[j-1]))
#                 else:
#                     graph.connect(_si[j], _si[j+1], calc_dist(_sx[j], _sy[j], _sx[j+1], _sy[j+1]))
#                     graph.connect(_si[j], _si[j-1], calc_dist(_sx[j], _sy[j], _sx[j-1], _sy[j-1]))
#     for i in range(len(edges)):
#         i_start = edges[i][0][0]
#         i_end  = edges[i][0][2]
#         for j in range(len(edges)):
#             if i == j:
#                 continue
#             j_start = edges[j][0][0]
#             j_end  = edges[j][0][2]
#             if i_start == j_start:
#                 graph.connect(edges[i][0], edges[j][0], 0)
#             if i_start == j_end:
#                 graph.connect(edges[i][0], edges[j][1], 0)
#             if i_end == j_start:
#                 graph.connect(edges[i][1], edges[j][0], 0)
#             if i_end == j_end:
#                 graph.connect(edges[i][1], edges[j][1], 0)
#     return graph

# idx = [1, 2, 4, 3, 1]
# x = [0, 0, 100, 100, 0]
# y = [50, 0, 0, 50, 50]
# split_length = 10

# graph = gen_graph(idx, x, y, split_length)

# print_shortest_path_bellman_ford(graph, '1-2_2', '3-4_1')
            
# fig = plt.figure(figsize=(10,10))
# ax1 = plt.subplot2grid((1,1), (0,0))
# ax1.plot(x, y, 'o-', linewidth=2, color='k')
# ax1.set_aspect('equal')
# ax1.grid(True)
# fig.show()

