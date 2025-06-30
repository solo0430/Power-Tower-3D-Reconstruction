import os
import sys
import time
import gmsh
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mesh_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("自适应网格生成器")

def create_adaptive_mesh(step_file, output_file=None, 
                         min_size=10.0, max_size=100.0, 
                         transition_factor=3.0, show_gui=False):
    """
    为STEP文件创建基础自适应网格
    
    参数:
        step_file: STEP文件路径
        output_file: 输出文件路径(如果不指定则基于输入文件名生成)
        min_size: 最小网格尺寸(用于小体积和曲率大的区域)
        max_size: 最大网格尺寸(用于大体积和平坦区域)
        transition_factor: 控制网格尺寸过渡平滑程度
        show_gui: 是否显示Gmsh图形界面
    """
    start_time = time.time()
    
    # 检查输入文件
    if not os.path.exists(step_file):
        logger.error(f"输入文件不存在: {step_file}")
        return False
    
    # 设置输出文件
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(step_file))[0]
        output_file = f"{base_name}_mesh.msh"
    
    # 初始化Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    try:
        logger.info(f"开始处理STEP文件: {step_file}")
        
        # 创建新模型
        model_name = os.path.basename(step_file)
        gmsh.model.add(model_name)
        
        # 导入STEP文件
        logger.info("正在导入STEP文件...")
        import_start = time.time()
        
        try:
            # 首选OCC导入(更稳定的方式)
            gmsh.model.occ.importShapes(step_file)
            gmsh.model.occ.synchronize()
            logger.info("使用OCC引擎导入成功")
        except:
            # 备选标准导入
            gmsh.open(step_file)
            logger.info("使用标准方式导入成功")
        
        import_time = time.time() - import_start
        logger.info(f"导入完成，耗时: {import_time:.2f}秒")
        
        # 获取所有体积实体
        volumes = gmsh.model.getEntities(3)
        logger.info(f"模型包含 {len(volumes)} 个体积实体")
        
        # 基础网格参数设置
        logger.info("配置网格生成参数...")
        
        # 设置全局网格大小参数
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
        
        # 启用自适应网格功能
        # 从几何曲率自动调整网格尺寸
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
        
        # 从边界扩展网格尺寸，实现平滑过渡
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        
        # 设置网格生成算法
        gmsh.option.setNumber("Mesh.Algorithm", 6)        # 二维前沿算法
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)      # 三维Delaunay算法
        
        # 适度的网格优化
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)   # 禁用Netgen优化以节省时间
        
        # 使用一阶元素，速度更快
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        
        # 适当放宽几何公差，提高稳定性
        gmsh.option.setNumber("Geometry.Tolerance", 1e-5)
        
        # 根据体积大小自适应调整网格尺寸
        logger.info("设置基于体积的自适应尺寸...")
        
        # 计算体积，用于确定网格尺寸
        volume_sizes = {}
        total_volume = 0
        
        for dim, tag in volumes:
            # 获取体积属性
            mass = gmsh.model.occ.getMass(dim, tag)
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            
            # 存储体积信息
            volume_sizes[tag] = mass
            total_volume += mass
            
            # 输出体积信息(对于大型模型仅输出部分信息)
            if len(volumes) < 20 or tag % 50 == 0:
                logger.info(f"体积 {tag}: 大小 = {mass:.1f}, 中心点 = ({com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f})")
        
        # 计算平均体积
        avg_volume = total_volume / len(volumes) if volumes else 0
        logger.info(f"模型总体积: {total_volume:.1f}, 平均体积: {avg_volume:.1f}")
        
        # 根据体积大小设置不同的网格尺寸
        for dim, tag in volumes:
            volume = volume_sizes[tag]
            
            # 计算网格尺寸: 体积小的用小网格，体积大的用大网格
            volume_ratio = volume / avg_volume
            size_factor = min(max(volume_ratio, 0.1), 10.0)  # 限制在0.1到10之间
            
            # 计算适应体积的网格尺寸
            mesh_size = min_size + (max_size - min_size) * size_factor / 10.0
            
            # 设置该体积的网格尺寸
            gmsh.model.mesh.setSize([(dim, tag)], mesh_size)
        
        # 创建距离场实现网格尺寸平滑过渡
        logger.info("设置网格平滑过渡...")
        
        try:
            # 识别小体积实体
            small_volumes = []
            for dim, tag in volumes:
                if volume_sizes[tag] < avg_volume * 0.2:  # 小于平均值20%的视为小体积
                    small_volumes.append(tag)
            
            if small_volumes:
                logger.info(f"检测到 {len(small_volumes)} 个小体积实体，为其创建过渡区域")
                
                # 获取小体积实体的表面
                small_faces = []
                for small_tag in small_volumes:
                    faces = gmsh.model.getBoundary([(3, small_tag)], combined=False)
                    small_faces.extend([abs(f[1]) for f in faces])
                
                # 创建距离场 - 计算到小体积实体表面的距离
                field_distance = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(field_distance, "FacesList", small_faces)
                
                # 创建阈值场 - 实现平滑过渡
                field_threshold = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(field_threshold, "IField", field_distance)
                gmsh.model.mesh.field.setNumber(field_threshold, "LcMin", min_size)
                gmsh.model.mesh.field.setNumber(field_threshold, "LcMax", max_size)
                gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", min_size * transition_factor)
                
                # 使用最小场合并所有字段
                field_min = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", [field_threshold])
                
                # 设置为背景场
                gmsh.model.mesh.field.setAsBackgroundMesh(field_min)
                logger.info("已设置网格过渡场")
        except Exception as e:
            logger.warning(f"创建过渡场失败，回退到基本自适应: {str(e)}")
        
        # 生成网格
        logger.info("开始生成网格...")
        mesh_start = time.time()
        
        try:
            # 先生成二维网格
            logger.info("生成表面网格...")
            gmsh.model.mesh.generate(2)
            
            # 然后生成三维网格
            logger.info("生成体积网格...")
            gmsh.model.mesh.generate(3)
            
            mesh_time = time.time() - mesh_start
            logger.info(f"网格生成成功，耗时: {mesh_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"网格生成失败: {str(e)}")
            
            # 尝试分步处理
            try:
                logger.info("尝试分步网格生成...")
                
                # 重设网格
                gmsh.model.mesh.clear()
                
                # 增加稳定性的设置
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)  # 禁用曲率自适应
                gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # 尝试不同的3D算法
                
                # 再次生成网格
                logger.info("生成表面网格(简化模式)...")
                gmsh.model.mesh.generate(2)
                
                logger.info("生成体积网格(简化模式)...")
                gmsh.model.mesh.generate(3)
                
                logger.info("简化模式网格生成成功")
                
            except Exception as e2:
                logger.error(f"简化模式网格生成也失败: {str(e2)}")
                return False
        
        # 保存网格文件
        logger.info(f"保存网格到: {output_file}")
        gmsh.write(output_file)
        
        # 获取并输出网格统计信息
        try:
            node_count = gmsh.model.mesh.getNodeCount()
            elem_count = gmsh.model.mesh.getElementCount()
            
            # 统计不同类型元素
            element_types = {}
            for dim in range(4):
                element_types[dim] = gmsh.model.mesh.getElementTypes(dim)
            
            # 输出统计信息
            logger.info(f"网格统计信息:")
            logger.info(f"- 节点数: {node_count}")
            logger.info(f"- 单元总数: {elem_count}")
            
            type_names = {
                2: "三角形",
                4: "四面体"
            }
            
            for dim, types in element_types.items():
                if not types:
                    continue
                dim_name = ["点", "线", "面", "体积"][dim]
                for type_idx in types:
                    type_name = type_names.get(type_idx, str(type_idx))
                    type_count = len(gmsh.model.mesh.getElementsByType(type_idx)[1])
                    logger.info(f"- {dim_name}单元 ({type_name}): {type_count}个")
                    
            total_time = time.time() - start_time
            logger.info(f"总处理时间: {total_time:.2f}秒")
            
        except Exception as e:
            logger.warning(f"获取网格统计信息失败: {str(e)}")
        
        # 显示GUI(如果需要)
        if show_gui:
            logger.info("启动GUI界面...")
            gmsh.fltk.run()
            
        return True
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        return False
        
    finally:
        # 清理资源
        gmsh.finalize()

def main():
    """主函数"""
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="STEP文件自适应网格划分工具")
    parser.add_argument("step_file", help="输入的STEP文件路径")
    parser.add_argument("-o", "--output", help="输出网格文件路径")
    parser.add_argument("--min-size", type=float, default=10.0, help="最小网格尺寸(默认: 10.0)")
    parser.add_argument("--max-size", type=float, default=100.0, help="最大网格尺寸(默认: 100.0)")
    parser.add_argument("--transition", type=float, default=3.0, help="网格过渡因子(默认: 3.0)")
    parser.add_argument("--gui", action="store_true", help="显示Gmsh图形界面")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用网格生成函数
    success = create_adaptive_mesh(
        args.step_file,
        args.output,
        args.min_size,
        args.max_size,
        args.transition,
        args.gui
    )
    
    # 返回适当的退出码
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
