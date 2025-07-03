import os
import sys
import numpy as np
import json
import math
import time  

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
import seaborn as sns

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
lib_direcotry = current_directory+"/../../../devel/lib"  # ugly but useful
sys.path.append(lib_direcotry)
print(lib_direcotry)

import pyCalib


def showpts(pts):
    # build a new matplotlib figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Scatter Plot')
    plt.show()
    
def projection(data):
    # Initial (x y z) Raw (x y z) intensity angle time
    xyz_l = data[:,3:6]
    rotor = data[:,7]
    
def cartesian_to_polar(normals):
    """Convert normalized cartesian coordinates to polar coordinates (theta, phi)"""
    theta = np.arccos(normals[:, 2])  # theta from z-axis [0, pi]
    phi = np.arctan2(normals[:, 1], normals[:, 0])  # phi in x-y plane [-pi, pi]
    return np.column_stack((theta, phi))

def visualize_normal_distribution(normals, title, output_path, before_after="before"):
    """Visualize normal vector distribution in both 3D and 2D polar projection"""
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(normals[:, 0], normals[:, 1], normals[:, 2], alpha=0.6, s=1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D Normal Distribution ({before_after} sampling)')
    
    ax2 = fig.add_subplot(122)
    polar_coords = cartesian_to_polar(normals)
    theta_edges = np.linspace(0, np.pi, 37)
    phi_edges = np.linspace(-np.pi, np.pi, 73)
    H, xedges, yedges = np.histogram2d(polar_coords[:, 0], polar_coords[:, 1],
                                      bins=[theta_edges, phi_edges])
    im = ax2.imshow(H.T, extent=[0, np.pi, -np.pi, np.pi], 
                    aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax2, label='Count')
    ax2.set_xlabel('θ (radians)')
    ax2.set_ylabel('φ (radians)')
    ax2.set_title(f'2D Polar Projection ({before_after} sampling)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    [功能描述]：激光雷达-旋转平台标定主函数，执行完整的标定流程包括数据加载、初始投影、优化标定和重加权优化
    @param 无参数
    @return 无返回值，处理结果保存为点云文件
    """

    # 设置原始数据路径并加载npz格式的标定数据
    rotlidarpath_npz = "/home/leo/workspace/data/motor_calibration_data/c.npz"
    rawdata = np.load(rotlidarpath_npz)  # 加载numpy压缩数组文件
    rawdata = rawdata['arr_0']           # 提取数组数据（包含点云坐标、强度、角度、时间等信息）
    
    # 步骤1：使用初始外参参数投影点云并保存原始结果
    # 初始参数：[绕Z轴旋转角, 绕Y轴旋转角, 绕X轴旋转角, X平移, Y平移, Z平移, 时间偏移]
    initial_params = [0,-math.radians(60), 0, 0, 0, 0, 0]  # 初始Y轴旋转角度为-60度
    pyCalib.ProjectionUsingCalibParam("/home/leo/workspace/data/motor_calibration_data/a.pcd",
                                      initial_params, rawdata[0:500000, :])  # 使用前50万个点进行投影
    print("Saved original point cloud: a.pcd")  # 输出保存信息
    
    # 步骤2：执行标定参数优化并记录执行时间
    start_time = time.perf_counter()  # 记录开始时间（高精度计时器）
    # 调用标定函数：初始参数、原始数据、优化次数(1次)、体素大小(1.0米)
    accurateValues = pyCalib.Calib(initial_params, rawdata[0:500000, :], 1, 1.0)
    elapsed = time.perf_counter() - start_time  # 计算优化耗时
    print("Calib returned parameters:", accurateValues)    # 输出优化后的参数
    print(f"Calib took {elapsed:.3f} seconds")            # 输出执行时间（保留3位小数）
    
    # 使用优化后的参数重新投影点云并保存精细标定结果
    pyCalib.ProjectionUsingCalibParam("/home/leo/workspace/data/motor_calibration_data/a_fine_v1_p0.3.pcd",
                                      accurateValues, rawdata[0:500000, :])
    print("Saved fine calibrated point cloud: a_fine.pcd")  # 输出保存信息
    
    # 步骤3：执行重加权优化（提高标定精度）并记录执行时间
    start_time = time.perf_counter()  # 重新记录开始时间
    # 调用重加权标定函数：初始参数、原始数据、优化次数(1次)、体素大小(2.0米)
    reweightedValues = pyCalib.Calib_reweight(initial_params, rawdata[0:500000, :], 1, 2.0)
    elapsed = time.perf_counter() - start_time  # 计算重加权优化耗时
    print("Calib_reweight returned parameters:", reweightedValues)  # 输出重加权后的参数
    print(f"Calib_reweight took {elapsed:.3f} seconds")           # 输出执行时间
    
    # 使用重加权优化后的参数投影点云并保存最终结果
    pyCalib.ProjectionUsingCalibParam("/home/leo/workspace/data/motor_calibration_data/c_fine_reweighted_v2.0_p0.7.pcd",
                                       reweightedValues, rawdata[0:500000, :])
    print("Saved reweighted calibrated point cloud: a_fine_reweighted.pcd")  # 输出保存信息

if __name__ == '__main__':
    main()
