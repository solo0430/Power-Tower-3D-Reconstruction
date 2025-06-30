# 🗼 Power Tower 3D Reconstruction for EM Simulation

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Power Systems](https://img.shields.io/badge/application-power%20systems-orange.svg)]()

**Complete open-source pipeline for reconstructing power towers from drone point clouds, optimized for electromagnetic simulation**

---

## 🎯 Problem Statement

Power system electromagnetic analysis faces a critical challenge: **lack of simulation-ready tower models**.

- 🚫 Existing models are designed for mechanical analysis, not EM simulation
- ⚠️ Severe self-intersection issues that break EM solvers  
- 🔧 Manual geometry repair is time-consuming and error-prone
- 📋 Limited availability for specific tower configurations

## 💡 Our Solution

**3-Stage Pipeline**: Real drone point clouds → Clean wireframes → EM simulation-ready solids

```
🚁 PLY Point Cloud → 📐 Wireframe Extraction → 🔧 Geometry Cleaning → 🔷 Solid Generation → 🔬 EM Simulation
```

## 🏗️ Architecture

```
src/ptwr/
├── ply_to_wireframe.py        # Stage 1: Point cloud → wireframe  
├── remove_specific_lines.py   # Stage 2: Wireframe cleaning
└── wireframe_to_solid.py      # Stage 3: Wireframe → solid model
```

## 🚀 Quick Start

### Installation

```
# Clone repository
git clone https://github.com/solo0430/Power-Tower-3D-Reconstruction.git
cd Power-Tower-3D-Reconstruction

# Install dependencies
pip install cadquery open3d scikit-learn matplotlib gmsh numpy
```

### Basic Usage

```
# Stage 1: Extract wireframe from point cloud
python src/ptwr/ply_to_wireframe.py tower_pointcloud.ply wireframe.step \
    --complexity 3 --voxel 0.02

# Stage 2: Clean wireframe (interactive GUI)
python src/ptwr/remove_specific_lines.py wireframe.step cleaned.step

# Stage 3: Generate solid model
python src/ptwr/wireframe_to_solid.py cleaned.step solid_tower.step \
    --width 0.05 --height 0.05 --shorten 0.05
```

### Python API

```
from ptwr.ply_to_wireframe import PointCloudToWireframe
from ptwr.wireframe_to_solid import WireframeToSolid

# Convert point cloud to wireframe
converter = PointCloudToWireframe(complexity_level=3)
converter.convert("tower.ply", "wireframe.step")

# Generate solid model
solid_gen = WireframeToSolid(width=0.05, height=0.05)
solid_gen.convert("wireframe.step", "tower_solid.step")
```

## 🔧 Key Features

### 🎯 Intelligent Point Cloud Processing
- **RANSAC-based line extraction** from noisy drone data
- **DBSCAN clustering** for automatic vertex deduplication  
- **5-level complexity control** for different tower types

### 🔷 Self-Intersection Prevention
- **Edge shortening algorithm** eliminates geometric conflicts
- **Adaptive rectangular extrusion** with controllable dimensions
- **Automatic fallback strategies** for complex geometries

### 🛠️ Commercial Software Ready
- Direct export to **STEP/STL** formats
- Compatible with **ANSYS HFSS, CST Studio, COMSOL**
- Maintains **watertight geometry** for reliable meshing

## 📊 Performance

| Stage | Typical Time | Input/Output |
|-------|-------------|--------------|
| Point Cloud Processing | 2-5 min | PLY → STEP wireframe |
| Wireframe Cleaning | 30s-2min | Manual interaction |
| Solid Generation | 1-3 min | STEP → Solid STEP |

**Tested on**: Standard transmission towers (20-50m height, 100k-500k points)

## 🎓 Applications

### Power System EM Analysis
- ⚡ Lightning protection design
- 📡 EMI/EMC compliance studies  
- 🔌 Insulation coordination
- 📊 Fault current distribution

### Infrastructure Monitoring
- 🚁 Automated tower inspection workflows
- 📏 Structural change detection
- 🗺️ Asset management integration

## 🔬 Technical Details

### Stage 1: Point Cloud → Wireframe
```
# Adaptive parameter tuning based on complexity
complexity_factors = {
    1: {'voxel_mult': 2.0, 'ransac_mult': 2.0},    # Simple towers
    3: {'voxel_mult': 1.0, 'ransac_mult': 1.0},    # Standard towers  
    5: {'voxel_mult': 0.5, 'ransac_mult': 0.5}     # Complex substations
}
```

### Stage 3: Self-Intersection Prevention
```
# Edge shortening to avoid intersections
shortened_length = original_length * (1 - 2 * shorten_ratio)
start_point = start + direction * (original_length * shorten_ratio)
end_point = end - direction * (original_length * shorten_ratio)
```

## 📚 Documentation

- [📖 **User Guide**](docs/user_guide.md) - Step-by-step tutorials
- [🔧 **API Reference**](docs/api_reference.md) - Function documentation  
- [❓ **FAQ**](docs/faq.md) - Common issues and solutions
- [🔬 **Technical Paper**](docs/technical_details.md) - Algorithm details

## 🤝 Contributing

We welcome contributions from the power systems and computer vision communities!

```
# Development setup
git clone https://github.com/solo0430/Power-Tower-3D-Reconstruction.git
cd Power-Tower-3D-Reconstruction
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

**Areas for contribution:**
- 🐛 Bug fixes and performance optimization
- 📊 New tower types and validation data
- 🔬 Advanced algorithms (ML-based line detection, etc.)
- 📚 Documentation and tutorials

## 📄 Citation

If this work helps your research, please cite:

```
@software{power_tower_3d_reconstruction,
  title={Power Tower 3D Reconstruction for EM Simulation},
  author={solo0430},
  year={2025},
  url={https://github.com/solo0430/Power-Tower-3D-Reconstruction},
  note={Open-source pipeline for drone point cloud to EM simulation models}
}
```

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [Open3D](http://www.open3d.org/) - Point cloud processing
- [CadQuery](https://cadquery.readthedocs.io/) - Parametric 3D modeling
- [FreeCAD](https://www.freecadweb.org/) - Open source CAD platform

---

## 🌟 **Making electromagnetic analysis of power systems more accessible and accurate**

*Bridging the gap between real-world infrastructure and simulation-ready models*
```
# Power-Tower-3D-Reconstruction
# Power-Tower-3D-Reconstruction
