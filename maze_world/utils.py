import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

def _setup_fig_ax(fig, ax, maze, start, goal, trajectory=None):
    # カラーマップの定義
    cmap = ListedColormap(['lightblue', 'brown', 'red', 'yellow'])
    processed_maze = np.zeros_like(maze)
    
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            processed_maze[i][j] = 0 if maze[i][j] == 0 else 1  # 道と障害物
    
    processed_maze[start] = 2  # スタート位置
    processed_maze[goal] = 3  # ゴール位置

    ax.set_title('Visualization of maze')
    ax.imshow(processed_maze, cmap=cmap)
    ax.axis('off')

    # 凡例のためのダミーデータ
    legend_elements = [Line2D([0], [0], color='brown', lw=4, label='Wall'),
                       Line2D([0], [0], color='lightblue', lw=4, label='Path')]

    if trajectory is not None:
        # 経路の描画
        for i in range(len(trajectory) - 1):
            ax.plot([trajectory[i][1], trajectory[i+1][1]], [trajectory[i][0], trajectory[i+1][0]], color='black', label='Trajectory' if i == 0 else "")
        legend_elements.append(Line2D([0], [0], color='black', lw=2, label='Trajectory'))

    ax.legend(handles=legend_elements, loc='upper right')

    return fig, ax
def visualize_maze_with_path(maze, start, goal, trajectory=None):
    plt.cla()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    _setup_fig_ax(fig, ax, maze, start, goal, trajectory)
    plt.show()


# スライドショー形式で迷路を表示する関数
def visualize_mazes_slideshow(mazes, start, goal, interval=1.0):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for maze in mazes:
        ax.cla()
        _setup_fig_ax(fig, ax, maze, start, goal)
        plt.pause(interval)  # 指定した時間間隔で待機
    plt.close('all')  # スライドショーの終了後に全ての図を閉じる