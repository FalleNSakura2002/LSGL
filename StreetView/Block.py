import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import shutil
import os

# 重置路径的方法
def chekdir(path):
    try:
        os.mkdir(path)
    except:
        print('文件夹已存在')

# 检查是否有结果
def chekres(path):
    if  len(os.listdir(path)) > 0:
        return 1
    else:
        return 0

# 获取网格
def Cut(center, length, coor_csv):
    # 重置输出目录
    path_coor_csv = os.path.dirname(coor_csv)
    chekdir(path_coor_csv)

    # 定义小正方形的边长和数量
    small_length = length / 10
    num_small_squares = 100

    # 计算所有小正方形的中心点坐标
    small_squares = []
    for i in range(num_small_squares):
        x = (i % 10) * small_length + center[0] - length / 2 + small_length / 2
        y = (i // 10) * small_length + center[1] - length / 2 + small_length / 2
        small_squares.append((i+1, x, y))

    # 输出小正方形的中心点坐标到CSV文件
    with open(coor_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'x', 'y'])
        writer.writerow(['0',center[0],center[1]])
        for row in small_squares:
            writer.writerow(row)

    # 绘制小正方形的中心点和周围的红色矩形
    fig, ax = plt.subplots()
    for square in small_squares:
        id, x, y = square
        ax.add_patch(Rectangle((x - small_length/2, y - small_length/2), small_length, small_length, linewidth=1, edgecolor='r', facecolor='none'))
        ax.plot(x, y, 'b.', markersize=10)

    # 设置图形属性
    ax.set_title('100 Small Squares')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    # 显示图形
    plt.show()
