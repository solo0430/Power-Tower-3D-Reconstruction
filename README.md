# 🗼 Power Tower 3D Reconstruction for EM Simulation

**Complete pipeline from drone point cloud to electromagnetic simulation-ready 3D models**

<p align="center">
  <img src="https://img.shields.io/badge/Field-Power%20Systems-blue" alt="Power Systems">
  <img src="https://img.shields.io/badge/Application-EM%20Simulation-green" alt="EM Simulation">
  <img src="https://img.shields.io/badge/Technology-Point%20Cloud-orange" alt="Point Cloud">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License">
</p>

## 🎯 Problem Statement

In power system electromagnetic analysis, there's a critical shortage of tower models suitable for EM simulation. Most existing models are designed for mechanical analysis and suffer from:

- **Self-intersection issues** that break EM solvers
- **Geometric inconsistencies** requiring extensive manual repair  
- **Incompatibility** with commercial simulation software
- **Limited availability** for specific tower configurations

## 💡 Our Solution

A complete **drone-to-simulation** pipeline that transforms real-world power towers into EM simulation-ready 3D models:

🚁 Drone Point Cloud → 📐 Line Extraction → 🗼 Wireframe Model → 🔷 Solid Model → 🔬 EM Simulation

## 🔧 Key Technologies

### **1. Intelligent Line Extraction**
- **RANSAC algorithm** for robust line fitting from noisy point clouds
- **DBSCAN clustering** for duplicate vertex removal
- **Adaptive complexity levels** for different tower types

### **2. Self-Intersection Prevention**
- **Edge shortening algorithm** to eliminate geometric conflicts
- **Intelligent union operations** for complex assemblies
- **Fallback strategies** for difficult geometries

### **3. Adaptive Mesh Generation**
- **Volume-based sizing** for optimal element distribution
- **Curvature-aware refinement** for accurate geometry capture
- **Smooth transition fields** for quality mesh gradients

## 🏗️ System Architecture

📁 src/
├── 🎯 point_cloud/
│ └── ply_to_wireframe.py # Point cloud → wireframe conversion
├── 🔷 wireframe/
│ └── wireframe_to_solid.py # Wireframe → solid model generation
└── 🕸️ mesh_generation/
└── adaptive_meshing.py # Automatic mesh generation

📁 examples/ # Sample data and usage examples
📁 docs/ # Technical documentation
📁 tests/ # Validation and test cases

## 🚀 Quick Start

### **Prerequisites**
pip install cadquery open3d scikit-learn matplotlib gmsh numpy


### **Basic Usage**
1. **Point Cloud to Wireframe**
python src/point_cloud/ply_to_wireframe.py tower.ply --output wireframe.step --complexity 3


2. **Wireframe to Solid**
python src/mesh_generation/adaptive_meshing.py solid.step --output mesh.msh --min-size 10 --max-size 100


## 📊 Results & Performance

### **Geometric Quality**
- ✅ **Zero self-intersections** in generated models
- ✅ **Smooth transitions** between tower components  
- ✅ **Preserved structural details** from original point cloud
- ✅ **Commercial software compatibility** (ANSYS, CST, COMSOL)

### **Processing Efficiency**  
- **Point cloud processing**: ~2-5 minutes for typical tower
- **Wireframe generation**: ~30 seconds
- **Solid model creation**: ~1-3 minutes
- **Mesh generation**: ~5-15 minutes (depending on complexity)

## 🎓 Applications

### **Electromagnetic Analysis**
- **Power line modeling** for EMI/EMC studies
- **Lightning protection** system design
- **Antenna placement** optimization on towers
- **RF propagation** analysis near power infrastructure

### **Power System Studies**
- **Insulation coordination** analysis
- **Switching transient** modeling  
- **Fault current distribution** visualization
- **Grounding system** optimization

## 🔬 Technical Innovation

### **Edge Shortening Algorithm**
Novel approach to prevent self-intersections by strategically shortening wireframe edges:
Shorten each edge by a percentage to avoid intersections

shorten_ratio = 0.05 # 5% from each end
new_length = original_length * (1 - 2 * shorten_ratio)


### **Adaptive Complexity Control**
Five-level complexity system for different tower types:
- **Level 1**: Simple transmission towers
- **Level 3**: Standard distribution towers  
- **Level 5**: Complex substation structures

### **Volume-Based Mesh Sizing**
Intelligent mesh generation based on geometric properties:
mesh_size = min_size + (max_size - min_size) * volume_factor


## 📚 Documentation

- 📖 **[User Guide](docs/user_guide.md)** - Complete usage instructions
- 🔧 **[API Reference](docs/api_reference.md)** - Detailed function documentation
- 🎯 **[Best Practices](docs/best_practices.md)** - Tips for optimal results
- 🐛 **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

## 🤝 Contributing

We welcome contributions from the power systems and computational geometry communities:

- 🐛 **Bug reports** and fixes
- ✨ **New features** and algorithms
- 📚 **Documentation** improvements
- 🧪 **Test cases** and validation data

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🏆 Citation

If you use this work in your research, please cite:

## 📚 Documentation

- 📖 **[User Guide](docs/user_guide.md)** - Complete usage instructions
- 🔧 **[API Reference](docs/api_reference.md)** - Detailed function documentation
- 🎯 **[Best Practices](docs/best_practices.md)** - Tips for optimal results
- 🐛 **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

## 🤝 Contributing

We welcome contributions from the power systems and computational geometry communities:

- 🐛 **Bug reports** and fixes
- ✨ **New features** and algorithms
- 📚 **Documentation** improvements
- 🧪 **Test cases** and validation data

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🏆 Citation

If you use this work in your research, please cite:

@software{power_tower_3d_reconstruction,
title={Power Tower 3D Reconstruction for EM Simulation},
author={[solo0430]},
year={2025},
url={https://github.com/solo0430/Power-Tower-3D-Reconstruction}
}


## 🔗 Related Work

- **Power system modeling**: IEEE standards for tower geometry
- **Point cloud processing**: Recent advances in LiDAR data analysis
- **Electromagnetic simulation**: Commercial solver compatibility requirements

---

**🌟 Bridging the gap between real-world infrastructure and simulation-ready models**

*Making electromagnetic analysis of power systems more accessible and accurate*
# Power-Tower-3D-Reconstruction
