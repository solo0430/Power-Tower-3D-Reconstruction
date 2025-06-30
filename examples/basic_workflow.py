#!/usr/bin/env python3
"""
Complete workflow example: Point cloud to simulation-ready model
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from point_cloud.ply_to_wireframe import PointCloudToWireframe
from wireframe.wireframe_to_solid import wireframe_to_solid_shortened
from mesh_generation.adaptive_meshing import create_adaptive_mesh

def complete_workflow(point_cloud_file, output_dir):
    """Complete processing pipeline"""
    
    # Step 1: Point cloud to wireframe
    wireframe_file = os.path.join(output_dir, "wireframe.step")
    converter = PointCloudToWireframe(complexity_level=3)
    converter.convert(point_cloud_file, wireframe_file)
    
    # Step 2: Wireframe to solid
    solid_file = os.path.join(output_dir, "solid.step")
    wireframe_to_solid_shortened(wireframe_file, solid_file, 
                                width=5.0, height=5.0, shorten_ratio=0.05)
    
    # Step 3: Generate mesh
    mesh_file = os.path.join(output_dir, "mesh.msh")
    create_adaptive_mesh(solid_file, mesh_file, 
                        min_size=10.0, max_size=100.0)
    
    print(f"✅ Complete workflow finished!")
    print(f"📁 Output files in: {output_dir}")

if __name__ == "__main__":
    # Example usage
    point_cloud_file = "sample_tower.ply" 
    output_dir = "results"
    
    os.makedirs(output_dir, exist_ok=True)
    complete_workflow(point_cloud_file, output_dir)
