import cadquery as cq
import numpy as np
import os

def wireframe_to_solid_shortened(input_file, output_file, width=5.0, height=5.0, shorten_ratio=0.05):
    """
    将线框STEP模型转换为具有矩形截面的实体模型，并缩短每条边的两端以减少自相交

    参数:
    - input_file: 输入的线框STEP文件路径
    - output_file: 输出的实体STEP文件路径
    - width: 矩形截面的宽度(mm)
    - height: 矩形截面的高度(mm)
    - shorten_ratio: 每条边缩短的比例(0-0.5之间)，如0.1表示两端各缩短10%
    """
    print(f"处理线框模型: {input_file}")

    # 验证缩短比例
    if shorten_ratio < 0 or shorten_ratio >= 0.5:
        print(f"警告: 缩短比例 {shorten_ratio} 无效，已设为默认值0.1")
        shorten_ratio = 0.05

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return False

    # 导入STEP文件
    try:
        model = cq.importers.importStep(input_file)
        print("模型导入成功")
    except Exception as e:
        print(f"模型导入失败: {str(e)}")
        return False

    # 提取所有边
    edges = model.edges().vals()
    print(f"找到 {len(edges)} 条边")

    if not edges:
        print("错误: 没有找到边，请确认模型是线框")
        return False

    # 创建列表保存所有实体
    solids = []

    # 处理每条边
    for i, edge in enumerate(edges):
        try:
            # 获取边的长度
            edge_length = edge.Length()

            if edge_length < 0.01:  # 跳过过短的边
                print(f"跳过过短的边 {i+1}")
                continue

            # 获取起点和终点
            start_point = edge.startPoint()
            end_point = edge.endPoint()

            # 计算方向向量
            direction = cq.Vector(
                end_point.x - start_point.x,
                end_point.y - start_point.y,
                end_point.z - start_point.z
            ).normalized()

            # 缩短边的两端
            shorten_amount = edge_length * shorten_ratio

            # 计算新的起点和终点
            new_start = cq.Vector(
                start_point.x + direction.x * shorten_amount,
                start_point.y + direction.y * shorten_amount,
                start_point.z + direction.z * shorten_amount
            )

            new_end = cq.Vector(
                end_point.x - direction.x * shorten_amount,
                end_point.y - direction.y * shorten_amount,
                end_point.z - direction.z * shorten_amount
            )

            # 计算缩短后的长度
            shortened_length = edge_length - 2 * shorten_amount

            if shortened_length < 0.01:  # 缩短后太短，跳过
                print(f"边 {i+1} 缩短后太短，已跳过")
                continue

            # 方法1: 沿边直接挤出矩形
            try:
                # 找到工作平面的垂直向量
                if abs(direction.z) < 0.9:  # 不与Z轴对齐
                    x_dir = direction.cross(cq.Vector(0, 0, 1)).normalized()
                else:  # 几乎与Z轴对齐，使用X轴
                    x_dir = direction.cross(cq.Vector(0, 1, 0)).normalized()

                y_dir = direction.cross(x_dir).normalized()

                # 在新起点创建工作平面
                plane = cq.Workplane(cq.Plane(
                    origin=new_start,
                    xDir=x_dir,
                    normal=direction
                ))

                # 创建矩形并沿缩短后的边长度挤出
                solid = plane.rect(width, height).extrude(shortened_length)
                solids.append(solid)
                print(f"处理边 {i+1}/{len(edges)} 使用挤出法（已缩短{shorten_ratio*100:.1f}%）")

            except Exception as e:
                print(f"挤出法处理边 {i+1} 失败: {str(e)}")

                # 方法2: 直接线条构造(备用方法)
                try:
                    # 在XY平面创建一条线
                    line = (cq.Workplane("XY")
                            .moveTo(0, 0)
                            .lineTo(shortened_length, 0))

                    # 创建矩形挤出体
                    box = line.rect(width, height).extrude(1)

                    # 旋转和定位以匹配原始边
                    # 计算从X轴到边方向的旋转
                    source = cq.Vector(1, 0, 0)
                    target = direction

                    # 获取旋转轴和角度
                    rotation_axis = source.cross(target)

                    if rotation_axis.Length < 1e-6:
                        # 向量平行，不需要旋转或围绕Y轴旋转180°
                        if source.dot(target) < 0:  # 相反方向
                            rotation_axis = cq.Vector(0, 1, 0)
                            angle = 180
                        else:
                            rotation_axis = None
                            angle = 0
                    else:
                        rotation_axis = rotation_axis.normalized()
                        cos_angle = source.dot(target)
                        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

                    # 应用旋转
                    if rotation_axis is not None:
                        box = box.rotate((0,0,0), (rotation_axis.x, rotation_axis.y, rotation_axis.z), angle)

                    # 移动到缩短后的起始位置
                    box = box.translate(new_start.x, new_start.y, new_start.z)

                    solids.append(box)
                    print(f"处理边 {i+1}/{len(edges)} 使用备用方法（已缩短{shorten_ratio*100:.1f}%）")

                except Exception as e2:
                    print(f"备用方法处理边 {i+1} 也失败: {str(e2)}")

        except Exception as e:
            print(f"处理边 {i+1} 时出错: {str(e)}")

    # 合并所有实体
    if solids:
        try:
            print(f"合并 {len(solids)} 个实体...")
            result = solids[0]
            for i, solid in enumerate(solids[1:]):
                print(f"联合实体 {i+2}/{len(solids)}")
                result = result.union(solid)

            # 导出最终模型
            print("导出最终模型...")
            cq.exporters.export(result, output_file)
            print(f"成功导出实体模型到: {output_file}")
            return True

        except Exception as e:
            print(f"使用union合并失败: {str(e)}")

            # 如果union失败，尝试导出为装配体
            try:
                print("尝试导出为装配体...")
                assembly = cq.Assembly()
                for i, solid in enumerate(solids):
                    assembly.add(solid, name=f"Part_{i+1}")

                assembly.save(output_file)
                print(f"已导出为装配体到: {output_file}")
                return True
            except Exception as e2:
                print(f"装配体导出也失败: {str(e2)}")

                # 最终备用方案: 导出单独的零件
                try:
                    output_dir = os.path.dirname(output_file)
                    base_name = os.path.splitext(os.path.basename(output_file))[0]

                    for i, solid in enumerate(solids):
                        part_file = os.path.join(output_dir, f"{base_name}_part_{i+1}.step")
                        cq.exporters.export(solid, part_file)

                    print(f"已导出 {len(solids)} 个独立零件文件到: {output_dir}")
                    return True
                except Exception as e3:
                    print(f"所有导出方法都失败: {str(e3)}")
                    return False
    else:
        print("没有创建有效的几何体")
        return False

if __name__ == "__main__":
    # 输入输出文件路径
    input_file = "/home/julia/3_wireframe.step"
    output_file = "/home/julia/3_solid_shortened.step"

    # 矩形截面尺寸(mm)
    width = 5.0
    height = 5.0

    # 边缩短比例 - 可以调整这个值来控制缩短程度
    # 0.1表示两端各缩短10%，共缩短20%
    shorten_ratio = 0.05

    # 执行转换
    wireframe_to_solid_shortened(input_file, output_file, width, height, shorten_ratio)
