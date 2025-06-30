#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云到线框转换模块 (PLY -> Wireframe STEP)
从无人机采集的杆塔点云中自动提取线段，生成可用于电磁仿真的线框模型

Author: [Your Name]
Date: 2025-07-01
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import time

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import cadquery as cq
    from cadquery import exporters
except ImportError:
    print("错误: 请安装 CadQuery: pip install cadquery")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PointCloudToWireframe:
    """点云到线框转换主类"""
    
    def __init__(self, 
                 voxel_size: float = 0.02,
                 ransac_distance: float = 0.015,
                 min_points_per_line: int = 120,
                 max_iterations: int = 1000,
                 min_line_length: float = 0.5,
                 complexity_level: int = 3):
        """
        初始化参数
        
        Args:
            voxel_size: 体素降采样尺寸
            ransac_distance: RANSAC拟合距离阈值
            min_points_per_line: 每条线段最小点数
            max_iterations: RANSAC最大迭代次数
            min_line_length: 最小线段长度过滤
            complexity_level: 复杂度等级 (1-5)
        """
        self.voxel_size = voxel_size
        self.ransac_distance = ransac_distance
        self.min_points_per_line = min_points_per_line
        self.max_iterations = max_iterations
        self.min_line_length = min_line_length
        self.complexity_level = complexity_level
        
        # 根据复杂度等级自适应调整参数
        self._adjust_parameters_by_complexity()
        
    def _adjust_parameters_by_complexity(self):
        """根据复杂度等级自适应调整参数"""
        complexity_factors = {
            1: {'voxel_mult': 2.0, 'ransac_mult': 2.0, 'min_points_mult': 0.5},
            2: {'voxel_mult': 1.5, 'ransac_mult': 1.5, 'min_points_mult': 0.7},
            3: {'voxel_mult': 1.0, 'ransac_mult': 1.0, 'min_points_mult': 1.0},
            4: {'voxel_mult': 0.7, 'ransac_mult': 0.7, 'min_points_mult': 1.3},
            5: {'voxel_mult': 0.5, 'ransac_mult': 0.5, 'min_points_mult': 1.5}
        }
        
        if self.complexity_level in complexity_factors:
            factor = complexity_factors[self.complexity_level]
            self.voxel_size *= factor['voxel_mult']
            self.ransac_distance *= factor['ransac_mult']
            self.min_points_per_line = int(self.min_points_per_line * factor['min_points_mult'])
            
        logger.info(f"复杂度等级 {self.complexity_level} - 调整后参数:")
        logger.info(f"  体素尺寸: {self.voxel_size:.4f}")
        logger.info(f"  RANSAC距离: {self.ransac_distance:.4f}")
        logger.info(f"  最小点数: {self.min_points_per_line}")

    def load_point_cloud(self, ply_path: Path) -> o3d.geometry.PointCloud:
        """
        加载并预处理点云
        
        Args:
            ply_path: PLY文件路径
            
        Returns:
            预处理后的点云
        """
        logger.info(f"加载点云文件: {ply_path}")
        
        if not ply_path.exists():
            raise FileNotFoundError(f"点云文件不存在: {ply_path}")
            
        # 加载点云
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if len(pcd.points) == 0:
            raise ValueError("点云文件为空或格式不正确")
            
        logger.info(f"原始点云包含 {len(pcd.points)} 个点")
        
        # 体素降采样
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        logger.info(f"降采样后包含 {len(pcd_downsampled.points)} 个点")
        
        # 统计离群点移除
        pcd_clean, ind = pcd_downsampled.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=2.0)
        logger.info(f"离群点移除后包含 {len(pcd_clean.points)} 个点")
        
        return pcd_clean

    def extract_line_segments_ransac(self, points: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        使用RANSAC迭代提取直线段
        
        Args:
            points: 点云数组 (N, 3)
            
        Returns:
            线段列表，每个元素为 (起点, 终点)
        """
        logger.info("开始RANSAC直线提取...")
        remaining_points = points.copy()
        line_segments = []
        iteration = 0
        
        while len(remaining_points) >= self.min_points_per_line and iteration < 100:
            iteration += 1
            logger.debug(f"RANSAC迭代 {iteration}, 剩余点数: {len(remaining_points)}")
            
            # 使用3D RANSAC拟合直线
            line_points = self._fit_3d_line_ransac(remaining_points)
            
            if line_points is None or len(line_points) < self.min_points_per_line:
                logger.debug("未找到足够的线段点，停止迭代")
                break
                
            # 计算线段端点
            start_point, end_point = self._calculate_line_endpoints(line_points)
            line_length = np.linalg.norm(end_point - start_point)
            
            # 过滤过短的线段
            if line_length >= self.min_line_length:
                line_segments.append((start_point, end_point))
                logger.debug(f"添加线段: 长度={line_length:.3f}m")
            else:
                logger.debug(f"跳过短线段: 长度={line_length:.3f}m")
                
            # 移除已处理的点
            remaining_points = self._remove_processed_points(remaining_points, line_points)
            
        logger.info(f"RANSAC提取完成，共找到 {len(line_segments)} 条线段")
        return line_segments

    def _fit_3d_line_ransac(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        3D空间中拟合直线
        
        Args:
            points: 点云数组
            
        Returns:
            拟合直线的内点，如果失败返回None
        """
        if len(points) < self.min_points_per_line:
            return None
            
        # 使用主成分分析确定主方向
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # SVD分解获取主方向
        _, _, vh = np.linalg.svd(centered_points)
        main_direction = vh[0]  # 第一主成分方向
        
        # 将点投影到主方向上
        projections = np.dot(centered_points, main_direction)
        
        # 使用投影值进行1D RANSAC
        ransac = RANSACRegressor(
            min_samples=self.min_points_per_line,
            residual_threshold=self.ransac_distance,
            max_trials=self.max_iterations,
            random_state=42
        )
        
        try:
            # 创建虚拟的X值（点在主方向上的投影）
            X = projections.reshape(-1, 1)
            y = np.zeros(len(projections))  # 虚拟y值
            
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
            
            if np.sum(inlier_mask) >= self.min_points_per_line:
                return points[inlier_mask]
            else:
                return None
                
        except Exception as e:
            logger.debug(f"RANSAC拟合失败: {e}")
            return None

    def _calculate_line_endpoints(self, line_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算线段的端点
        
        Args:
            line_points: 属于同一直线的点集
            
        Returns:
            (起点, 终点)
        """
        # 计算主方向
        centroid = np.mean(line_points, axis=0)
        centered = line_points - centroid
        _, _, vh = np.linalg.svd(centered)
        direction = vh[0]
        
        # 投影到主方向
        projections = np.dot(centered, direction)
        min_proj, max_proj = np.min(projections), np.max(projections)
        
        # 计算端点
        start_point = centroid + min_proj * direction
        end_point = centroid + max_proj * direction
        
        return start_point, end_point

    def _remove_processed_points(self, all_points: np.ndarray, processed_points: np.ndarray) -> np.ndarray:
        """
        从点集中移除已处理的点
        
        Args:
            all_points: 所有点
            processed_points: 已处理的点
            
        Returns:
            剩余的点
        """
        # 使用DBSCAN进行点匹配和移除
        if len(processed_points) == 0:
            return all_points
            
        # 为每个已处理点找到最近的原始点并移除
        remaining_mask = np.ones(len(all_points), dtype=bool)
        
        for proc_point in processed_points:
            distances = np.linalg.norm(all_points - proc_point, axis=1)
            closest_idx = np.argmin(distances)
            if distances[closest_idx] < self.ransac_distance * 2:
                remaining_mask[closest_idx] = False
                
        return all_points[remaining_mask]

    def remove_duplicate_vertices(self, line_segments: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        使用DBSCAN聚类去除重复顶点
        
        Args:
            line_segments: 原始线段列表
            
        Returns:
            去重后的线段列表
        """
        logger.info("开始DBSCAN顶点去重...")
        
        if not line_segments:
            return line_segments
            
        # 收集所有顶点
        all_vertices = []
        for start, end in line_segments:
            all_vertices.extend([start, end])
        all_vertices = np.array(all_vertices)
        
        # DBSCAN聚类
        eps = self.ransac_distance * 2  # 聚类半径
        clustering = DBSCAN(eps=eps, min_samples=1).fit(all_vertices)
        labels = clustering.labels_
        
        # 计算每个聚类的中心点
        unique_labels = np.unique(labels)
        cluster_centers = {}
        for label in unique_labels:
            if label != -1:  # 忽略噪声点
                cluster_points = all_vertices[labels == label]
                cluster_centers[label] = np.mean(cluster_points, axis=0)
        
        # 重新构建线段
        cleaned_segments = []
        vertex_idx = 0
        for start, end in line_segments:
            # 找到起点和终点对应的聚类中心
            start_label = labels[vertex_idx]
            end_label = labels[vertex_idx + 1]
            vertex_idx += 2
            
            if start_label in cluster_centers and end_label in cluster_centers:
                new_start = cluster_centers[start_label]
                new_end = cluster_centers[end_label]
                
                # 检查线段长度
                if np.linalg.norm(new_end - new_start) >= self.min_line_length:
                    cleaned_segments.append((new_start, new_end))
        
        logger.info(f"去重完成: {len(line_segments)} -> {len(cleaned_segments)} 条线段")
        return cleaned_segments

    def create_wireframe_step(self, line_segments: List[Tuple[np.ndarray, np.ndarray]]) -> cq.Compound:
        """
        将线段转换为CadQuery线框
        
        Args:
            line_segments: 线段列表
            
        Returns:
            CadQuery线框复合体
        """
        logger.info("构建CadQuery线框...")
        
        if not line_segments:
            raise ValueError("没有线段可以转换")
            
        edges = []
        for i, (start, end) in enumerate(line_segments):
            try:
                # 创建CadQuery向量
                start_vec = cq.Vector(float(start[0]), float(start[1]), float(start[2]))
                end_vec = cq.Vector(float(end[0]), float(end[1]), float(end[2]))
                
                # 创建边
                edge = cq.Edge.makeLine(start_vec, end_vec)
                edges.append(edge)
                
            except Exception as e:
                logger.warning(f"创建第{i+1}条边失败: {e}")
                continue
        
        if not edges:
            raise ValueError("无法创建任何有效边")
            
        # 创建复合体
        wireframe = cq.Compound.makeCompound(edges)
        logger.info(f"线框创建完成，包含 {len(edges)} 条边")
        
        return wireframe

    def visualize_results(self, 
                         original_points: np.ndarray, 
                         line_segments: List[Tuple[np.ndarray, np.ndarray]],
                         save_path: Optional[Path] = None):
        """
        可视化结果
        
        Args:
            original_points: 原始点云
            line_segments: 提取的线段
            save_path: 保存路径（可选）
        """
        fig = plt.figure(figsize=(15, 5))
        
        # 原始点云
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
                   c='blue', s=1, alpha=0.6)
        ax1.set_title('原始点云')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # 提取的线段
        ax2 = fig.add_subplot(132, projection='3d')
        for i, (start, end) in enumerate(line_segments):
            ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    'r-', linewidth=2, alpha=0.8)
        ax2.set_title(f'提取的线段 ({len(line_segments)} 条)')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        
        # 叠加显示
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
                   c='lightblue', s=1, alpha=0.3, label='原始点云')
        for i, (start, end) in enumerate(line_segments):
            ax3.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    'r-', linewidth=2, alpha=0.8)
        ax3.set_title('叠加显示')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化结果保存至: {save_path}")
        else:
            plt.show()

    def convert(self, 
                ply_path: Path, 
                output_step: Path,
                visualize: bool = True,
                save_visualization: bool = False) -> None:
        """
        执行完整的转换流程
        
        Args:
            ply_path: 输入PLY文件路径
            output_step: 输出STEP文件路径
            visualize: 是否显示可视化
            save_visualization: 是否保存可视化图片
        """
        start_time = time.time()
        
        try:
            # 1. 加载点云
            pcd = self.load_point_cloud(ply_path)
            points = np.asarray(pcd.points)
            
            # 2. 提取线段
            line_segments = self.extract_line_segments_ransac(points)
            
            if not line_segments:
                raise ValueError("未能从点云中提取到任何线段")
            
            # 3. 去重
            line_segments = self.remove_duplicate_vertices(line_segments)
            
            # 4. 创建线框
            wireframe = self.create_wireframe_step(line_segments)
            
            # 5. 导出STEP
            output_step.parent.mkdir(parents=True, exist_ok=True)
            exporters.export(wireframe, str(output_step))
            logger.info(f"线框STEP文件已保存: {output_step}")
            
            # 6. 可视化
            if visualize:
                viz_path = None
                if save_visualization:
                    viz_path = output_step.parent / f"{output_step.stem}_visualization.png"
                self.visualize_results(points, line_segments, viz_path)
            
            # 统计信息
            elapsed_time = time.time() - start_time
            logger.info(f"转换完成！用时: {elapsed_time:.2f}秒")
            logger.info(f"统计信息:")
            logger.info(f"  输入点数: {len(points)}")
            logger.info(f"  输出线段: {len(line_segments)}")
            logger.info(f"  平均线段长度: {np.mean([np.linalg.norm(end - start) for start, end in line_segments]):.3f}m")
            
        except Exception as e:
            logger.error(f"转换失败: {e}")
            raise


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description="将PLY点云转换为线框STEP文件")
    parser.add_argument("input_ply", type=Path, help="输入PLY文件路径")
    parser.add_argument("output_step", type=Path, help="输出STEP文件路径")
    parser.add_argument("--voxel", type=float, default=0.02, help="体素降采样尺寸 (默认: 0.02)")
    parser.add_argument("--ransac_dist", type=float, default=0.015, help="RANSAC距离阈值 (默认: 0.015)")
    parser.add_argument("--min_points", type=int, default=120, help="每条线最小点数 (默认: 120)")
    parser.add_argument("--min_length", type=float, default=0.5, help="最小线段长度 (默认: 0.5)")
    parser.add_argument("--complexity", type=int, default=3, choices=range(1, 6), 
                       help="复杂度等级 1-5 (默认: 3)")
    parser.add_argument("--no-viz", action="store_true", help="禁用可视化")
    parser.add_argument("--save-viz", action="store_true", help="保存可视化图片")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 创建转换器
    converter = PointCloudToWireframe(
        voxel_size=args.voxel,
        ransac_distance=args.ransac_dist,
        min_points_per_line=args.min_points,
        min_line_length=args.min_length,
        complexity_level=args.complexity
    )
    
    # 执行转换
    try:
        converter.convert(
            ply_path=args.input_ply,
            output_step=args.output_step,
            visualize=not args.no_viz,
            save_visualization=args.save_viz
        )
        print(f"\n✅ 转换成功！输出文件: {args.output_step}")
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
