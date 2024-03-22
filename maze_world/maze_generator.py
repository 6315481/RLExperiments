import numpy as np
from utils import visualize_maze_with_path, visualize_mazes_slideshow
from collections import namedtuple
import matplotlib.pyplot as plt

class UnionFind:
    def __init__(self, islands):
        self.ancestor = {}
        for island in islands:
            self.ancestor[island] = island
    
    def root(self, v1):
        cur_v = v1
        while cur_v != self.ancestor[cur_v]:
            cur_v = self.ancestor[cur_v]
        return cur_v
    
    def same(self, v1, v2):
        return self.root(v1) == self.root(v2)
    
    def unite(self, v1, v2):
        root_1, root_2 = self.root(v1), self.root(v2)
        if root_1 == root_2:
            return
        self.ancestor[root_2] = root_1

# Kruskal Algorithm
def _calculate_mst(islands):
    Edge = namedtuple("Edge", ["src", "to", "dist"])
    edges = [Edge((ia, ja), (ib, jb), abs(ia - ib) + abs(ja - jb))  for ia, ja in islands for ib, jb in islands]
    edges.sort(key=lambda item: item.dist)
    
    uf = UnionFind(islands)

    mst_edges = []
    for edge in edges:
        if not uf.same(edge.src, edge.to):
            uf.unite(edge.src, edge.to)
            mst_edges.append(edge)

    return mst_edges

def _join_islands(maze, mst_edges):
    for edge in mst_edges:
        (isr, jsr), (it, jt) = edge.src, edge.to
        sign_i, sign_j = int(np.sign(it - isr)), int(np.sign(jt - jsr))
        for i in range(abs(it - isr)):
            cur_i = isr + sign_i * i
            maze[cur_i][jsr] = 0
        for j in range(abs(jt - jsr)):
            cur_j = jsr + sign_j * j
            maze[it][cur_j] = 0

def maze_generator(width, height, num_islands=70):
    maze = [[1 for j in range(width)] for i in range(height)]
    start = (0, 0)
    goal = (height-1, width-1)
    islands = [start, goal]
    candidates = [(i, j) for i in range(height) for j in range(width) if (i, j) != start and (i, j) != goal]
    islands = islands + [candidates[i] for i in np.random.choice(range(len(candidates)), num_islands, replace=False)]
    for i, j in islands:
        maze[i][j] = 0
    
    mst_edges = _calculate_mst(islands)
    _join_islands(maze, mst_edges)

    return maze

if __name__ == "__main__":
    mazes = []
    for i in range(20):
        width, height = 50, 50
        start = (0, 0)
        goal = (height-1, width-1)
        mazes.append(maze_generator(width, height, num_islands=260))

        visualize_mazes_slideshow(mazes, start, goal)