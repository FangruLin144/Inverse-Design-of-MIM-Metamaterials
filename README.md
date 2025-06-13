# Adjoint-Based Optimization Framework for MIM Metasurface Absorbers



## Future Work Directions

### 0. Remove Any Known Bugs

For example, negative thicknesses and boundary clipping coherent with physical constraints.

### 1. Modularize Optimizer Logic with Full Autograd Support

This includes: 
- Replace optax with a pure autograd-compatible optimizer (e.g., SGD or Adam variants you write or wrap)
- Cleanly separate optimizer logic into a new optimizer.py
- Provide a flexible optimizer interface (pluggable optimizer)

### 2. Logging and History Management

The current framework stores simulation and flux histories. A future work could include a more structured **result management system**, which automatically saves:
- design parameters
- losses
- reflectance curves
- full simulation outputs (HDF5, JSON)

Preferrably utilities can be added to: 
- Resume from intermediate checkpoints
- Compare different optimization runs
- Export for plotting outside Python

Need to also come up with result sharing mechanisms across the group.

An example structure of the result management system could look like: 

```
results/
├── run_2025-06-13_14-30-12/         # Unique ID based on timestamp or task hash
│   ├── metadata.json                # Human- and machine-readable metadata
│   ├── params_history.npy           # List of design parameter arrays per iteration
│   ├── error_history.npy            # Scalar loss values per iteration
│   ├── fluxes_history.npy           # Flux spectrum per iteration
│   ├── reflectance_curves/          # Saved reflectance curves (one per iteration)
│   │   ├── iter_000.png
│   │   ├── iter_001.png
│   │   └── ...
│   ├── sim_output/                  # Optional full Tidy3D outputs (.hdf5), this could be quite large 
│   │   ├── flux_0.hdf5
│   │   └── ...
│   └── checkpoint.pkl               # To resume optimization mid-run
│
├── run_2024-06-14_16-30-42/
...
```

### 3. Refactor into a Professional Python Package

Currently, everything lives in one notebook. The objectives of a professionally packaged code: 
- Make collaboration easier
- Enable testing individual modules
- Allow deployment from command line or batch scripts

An example structure of the framework could look like: 

```
meta_atom_optimizer/
├── src/                    # Core source code modules
│   ├── __init__.py             # Package initializer
│   ├── structures.py           # Parametric definition of layered geometries (substrate + cuboid stack)
│   ├── simulator.py            # Builds Tidy3D simulation object with sources, monitors and mesh
│   ├── objective.py            # Target spectrum design, loss functions and objective evaluation
│   ├── optimize.py             # Optimization loop with autograd-compatible gradient updates
│   ├── utils.py                # Helper functions for visualization, reshaping and logging
│   └── config.py               # Global constants and constraint settings (wavelengths, material bounds, etc.)
│
├── notebooks/
│   └── main.ipynb              # Main interactive notebook for prototyping and visualization
│
├── data/                       # Folder for raw Tidy3D outputs and downloaded simulation data (.hdf5)
│
├── tests/
│   └── test_structure.py       # Unit tests for structure generation and constraint checking
│
├── requirements.txt            # Python dependencies required to run the framework
├── README.md                   # Project overview, usage and documentation
├── LICENSE                     # Project license (e.g., MIT)
└── .gitignore                  # Files and folders excluded from version control
```

### 4. Add Multi-Objective Support (e.g., Bandwidth + Impedance Matching)

The current objective function can be naturally extended to simultaneously optimize for:
- Broadband absorption, and
- Impedance matching (e.g., by minimizing reflection at normal incidence or matching field profiles)

This requires minimal structural change: just update `objective_fn()` to combine multiple loss terms.

### 5. More Flexibility in Geometry Update 
