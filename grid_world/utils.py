import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# 迷路の可視化関数（凡例付き）
def visualize_maze_with_path(grid, start, goal, trajectory=None):
    # カラーマップの定義
    cmap = ListedColormap(['lightblue', 'brown', 'red', 'yellow'])
    maze = np.zeros_like(grid)
    
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            maze[i][j] = 0 if grid[i][j] == 0 else 1  # 道と障害物
    
    maze[start] = 2  # スタート位置
    maze[goal] = 3  # ゴール位置

    plt.figure(figsize=(10, 6))
    plt.title('Visualization of maze')
    plt.imshow(maze, cmap=cmap)
    plt.axis('off')

    # 凡例のためのダミーデータ
    legend_elements = [Line2D([0], [0], color='brown', lw=4, label='Wall'),
                       Line2D([0], [0], color='lightblue', lw=4, label='Path')]

    if trajectory is not None:
        # 経路の描画（ダミーの凡例を追加するためにlabel引数を使用）
        for i in range(len(trajectory) - 1):
            line = plt.plot([trajectory[i][1], trajectory[i+1][1]], [trajectory[i][0], trajectory[i+1][0]], color='black', label='trajectory' if i == 0 else "")
        legend_elements.append(Line2D([0], [0], color='black', lw=2, label='Trajectory'))

    plt.legend(handles=legend_elements, loc='upper right')
    plt.show()