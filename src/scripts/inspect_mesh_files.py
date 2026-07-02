"""Fingerprint HDF5 files to find the EIT ring mesh + mapping matrices (PLAN.md §14).

The EIT-reconstruction track (PLAN.md §3.4) needs three artifacts exported from the
MATLAB / Python EIT solver, but a data directory usually mixes them with unrelated
HDF5 (subject recordings, phantom validation meshes). This script opens every ``.h5``
under a directory and classifies it, so you can hand over the right files without
guessing:

  * MESH        - an EIT FEM mesh (has ``nodes`` / ``elems`` / ``elecs``); prints the
                  electrode count and node/element sizes. The ring is the one whose
                  ``num_elecs`` matches the ring (expected 32) and whose name is *not*
                  tank/resistor/pipette (those are validation phantoms).
  * MAPPING     - coarse->fine / mesh->image maps (``mat2D/c2f`` and/or ``mat2D/m2i``),
                  needed to turn the element-space Jacobian into a pixel-space operator
                  for ``src/models/eit_recon.py::EITForwardOperator``.
  * RECORDING   - a subject session (``data/bp``, ``data/pviHP`` ...): not needed here.
  * OTHER/ERROR - anything else, or a file that failed to open.

Usage (read-only; opens files, never writes):

    python -m src.scripts.inspect_mesh_files /path/to/h5_dir
    python -m src.scripts.inspect_mesh_files /path/to/dir --recursive --ring-elecs 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


def _has(group, path: str) -> bool:
    try:
        return path in group
    except Exception:
        return False


def classify(path: Path, ring_elecs: int) -> dict:
    info: dict = {"path": str(path), "kind": "OTHER", "detail": ""}
    try:
        with h5py.File(path, "r") as f:
            keys = set(f.keys())

            # --- EIT FEM mesh (see pvi_mesh2d.load_mesh_hdf5) ---
            if {"nodes", "elems"} <= keys:
                num_elecs = None
                if _has(f, "elecs/num_elecs"):
                    try:
                        num_elecs = int(f["elecs/num_elecs"][()])
                    except Exception:
                        num_elecs = None
                n_nodes = f["nodes"].shape[-1] if "nodes" in f else "?"
                n_elems = f["elems"].shape[-1] if "elems" in f else "?"
                info["kind"] = "MESH"
                is_ring = (num_elecs == ring_elecs) and not any(
                    tag in path.name.lower() for tag in ("tank", "resistor", "pipette")
                )
                info["detail"] = (
                    f"num_elecs={num_elecs} nodes={n_nodes} elems={n_elems}"
                    + ("  <-- likely RING" if is_ring else "  (phantom/other)")
                )
                info["ring_candidate"] = bool(is_ring)
                return info

            # --- coarse<->fine / mesh->image mapping matrices (pvi_inverse) ---
            maps = [m for m in ("c2f", "m2i", "i2f") if _has(f, f"mat2D/{m}")]
            if maps:
                info["kind"] = "MAPPING"
                info["detail"] = "mat2D/" + ", mat2D/".join(maps)
                return info

            # --- subject recording (not needed for the EIT track) ---
            if _has(f, "data/bp") or _has(f, "data/pviHP") or "masks" in keys:
                info["kind"] = "RECORDING"
                info["detail"] = "subject session (data/bp, data/pvi*)"
                return info

            info["detail"] = "top-level keys: " + ", ".join(sorted(keys))[:80]
    except Exception as exc:  # not HDF5, corrupt, permissions, ...
        info["kind"] = "ERROR"
        info["detail"] = f"{type(exc).__name__}: {exc}"
    return info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", help="Directory to scan for .h5 files.")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    p.add_argument("--ring-elecs", type=int, default=32,
                   help="Electrode count that identifies the ring mesh (default 32).")
    args = p.parse_args(argv)

    root = Path(args.directory)
    if not root.exists():
        print(f"[inspect_mesh_files] not found: {root}", file=sys.stderr)
        return 2

    files = sorted(root.rglob("*.h5") if args.recursive else root.glob("*.h5"))
    if not files:
        print(f"[inspect_mesh_files] no .h5 files under {root}", file=sys.stderr)
        return 1

    results = [classify(f, args.ring_elecs) for f in files]

    width = max((len(r["path"]) for r in results), default=10)
    for r in results:
        print(f"{r['kind']:<9} {r['path']:<{width}}  {r['detail']}")

    meshes = [r for r in results if r["kind"] == "MESH"]
    ring = [r for r in meshes if r.get("ring_candidate")]
    mappings = [r for r in results if r["kind"] == "MAPPING"]
    print("\n[summary]")
    print(f"  meshes:          {len(meshes)}  (ring candidates: {len(ring)})")
    print(f"  mapping files:   {len(mappings)}")
    if ring:
        print("  -> likely ring mesh(es):")
        for r in ring:
            print(f"       {r['path']}  ({r['detail'].split('  ')[0]})")
    if mappings:
        print("  -> mapping file(s) (for element->pixel operator):")
        for r in mappings:
            print(f"       {r['path']}  ({r['detail']})")
    if not ring:
        print("  -> no obvious ring mesh; re-run with --ring-elecs set to the ring's "
              "electrode count, or hand over the mesh whose num_elecs matches the ring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
