# Quantum Graphs (Python)

This repository provides Python classes and utilities for modeling **quantum graphs** — networks of one-dimensional bonds connected at vertices with scattering matrices.  

It includes:  

- **ClosedGraphs** (no leads)  
- **OpenGraphs** (with external leads)  
- **Wigner tools** for Random Matrix Theory (RMT) statistics  

---

## 📂 Repository Structure

    ClosedGraphs/
    ├── ClosedGraphs.py          # Closed quantum graph class
    └── examples/                # Example configurations
        ├── Graph1/
        ├── Graph2/
        ├── SingleRing/
        ├── DoubleRings/
        ├── MixingRings/
        ├── Pentagon/
        └── Tetrahedron/

    OpenGraphs/
    ├── OpenGraphs.py            # Open quantum graph class
    └── examples/
        ├── Graph1/
        ├── Graph2/
        └── RingGraph/

    wigner/
    ├── sigma2_RMT.py            # Number variance Σ²(L) for GOE, GUE, GSE, Poisson
    ├── delta3_RMT.py            # Spectral rigidity Δ₃(L)
    ├── wigner_long_range_correlations.py
    └── *.txt / *.npz            # Precomputed RMT curves and test data

---

## 🚀 Features

### **ClosedGraphs**  
- Construction of closed quantum graphs (multiple bonds and loops handled automatically)  
- Eigenvalue spectrum computation using σ_min method  
- Wavefunction calculation and visualization  
- Spectral statistics: staircase function, nearest-neighbor spacing (NNSD), Σ²(L), Δ₃(L)  
- Length spectrum and entropy analysis  

### **OpenGraphs**  
- Extension to graphs with external leads  
- Automatic handling of vertex scattering matrices with open ports  
- Transmission, reflection, and energy distribution plots  

### **Wigner tools**  
- Reference RMT predictions for **short- and long-range correlations**  
- Wigner surmise distributions (Poisson, GOE, GUE, GSE)  
- Benchmarking of experimental and numerical spectra against RMT  

---

## ⚡ Quick Start

1. **Clone the repository**

        git clone https://github.com/hugogirin/quantum-graphs.git
        cd quantum-graphs

2. **Run a minimal closed graph example**

        from ClosedGraphs.ClosedGraphs import ClosedGraphs

        # Triangle with 3 vertices
        vertex_positions = [(0,0), (1,0), (0.5,0.866)]
        port_map = [
            [(1,0), (2,0)],   # vertex 0 connects to v1 and v2
            [(0,0), (2,1)],   # vertex 1 connects to v0 and v2
            [(0,1), (1,1)]    # vertex 2 connects to v0 and v1
        ]

        graph = ClosedGraphs(vertex_positions, port_map, neff=2.45)

        # Compute eigenvalues
        spectrum, _ = graph.compute_spectrum(N_scan=200000)

        # Plot staircase function with Weyl law
        graph.plot_staircase_function(plot_weyl=True, save_fig=False)

3. **Explore the `examples/` folders** for ready-to-run configurations (Graph1, Pentagon, RingGraph…).  

---

## 📊 Using the Wigner Tools

The `wigner/` folder provides functions to compare your spectra with **RMT predictions** using **interpolated reference curves**.  
In particular, `wigner/delta3_RMT.py` exposes:

- `delta3_RMT(L, wigner="GOE")` → returns the spectral rigidity Δ₃(L) for `L` (scalar or array) and ensemble in `{"GOE","GUE","GSE"}`.

**Example: compute and plot Δ₃(L) for GOE, GUE, and GSE**

    import numpy as np
    from wigner.delta3_RMT import delta3_RMT

    # L grid
    L_vals = np.linspace(0, 20, 5000)

    # Interpolated Δ₃(L) from precomputed RMT tables
    delta3_goe = delta3_RMT(L_vals, wigner="GOE")
    delta3_gue = delta3_RMT(L_vals, wigner="GUE")
    delta3_gse = delta3_RMT(L_vals, wigner="GSE")

    # (Optional) quick plot
    import matplotlib.pyplot as plt
    plt.plot(L_vals, delta3_goe, label="GOE")
    plt.plot(L_vals, delta3_gue, label="GUE")
    plt.plot(L_vals, delta3_gse, label="GSE")
    plt.xlabel("L"); plt.ylabel("Δ₃(L)"); plt.legend(); plt.tight_layout(); plt.show()

Notes:
- The module expects the reference files (e.g., `delta3goe.txt`, `delta3gue.txt`, `delta3_RP_model_N65536_gamma_0.9_GSE.txt`) to be present in `wigner/`.  
  If your paths differ, edit the `working_dir` variable inside `wigner/delta3_RMT.py`.
- A similar API is provided in `sigma2_RMT.py` for the **number variance** Σ²(L) using interpolated reference curves.

---

## 📘 Documentation

- Each class (`ClosedGraphs`, `OpenGraphs`) is fully documented with parameters and methods.  
- Example configurations illustrate typical usage.  
- The `wigner` utilities provide direct comparison to Random Matrix Theory results.  

---

## 🧑‍💻 Author

**Hugo Girin**  
Centre de Nanosciences et de Nanotechnologies (C2N), Université Paris-Saclay  
_PhD research on quantum graphs and integrated photonics._  

---

## 📜 License

MIT License — free to use, modify, and distribute with attribution.  
