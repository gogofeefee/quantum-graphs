# Data used by the examples

## `mpb/shallow_etched_si_on_sio2_w0.40_h0.22_rib0.08/`

Dispersion of the fundamental TE-like mode (band 1) of a shallow-etched silicon
rib waveguide on SiO₂, computed with [MPB](https://mpb.readthedocs.io):

| Parameter        | Value            |
|------------------|------------------|
| Core / cladding  | Si (3.48) / SiO₂ (1.44) |
| Width            | 0.40 µm          |
| Height           | 0.22 µm          |
| Rib (slab) height| 0.084 µm         |

| File | Keys | Meaning |
|------|------|---------|
| `neff_vs_kvac_band1_full.npz`        | `k_vac`, `neff` | effective index vs vacuum wavenumber k = 2π/λ (µm⁻¹) |
| `group_index_vs_kvac_band1_full.npz` | `k_vac`, `ng`   | group index vs vacuum wavenumber (µm⁻¹) |

They are loaded by the `config_nfunc*.py` examples, which interpolate them into
callables `neff(k)` and `ng(k)` (see `load_mpb_indices`).
