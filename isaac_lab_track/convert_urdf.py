#!/usr/bin/env python3
"""
Convert iRonCub-Mk1_1 URDF → USD for Isaac Lab 2.x.

Isaac Lab 2.x removed the old `scripts/tools/convert_urdf.py` CLI.
This script uses the `URDFParseAndImportFile` Omniverse Kit command directly.

Key preprocessing steps:
  1. Strip <gazebo> and top-level <sensor> elements — the v2 parser crashes on them.
  2. Strip <visual> elements — only collision geometry is needed for physics.
  3. Rewrite collision mesh filenames: hyphens → underscores in STL names
     (e.g. sim_sea_2-5_root_link_prt.stl → sim_sea_2_5_root_link_prt.stl)
     and create symlinks so the files can be found at those sanitized paths.

Output: assets/iRonCub/robots/iRonCub-Mk1_1/ironcub.usd

Run once per machine from the repo root:
    OMNI_KIT_ACCEPT_EULA=Y conda run -n polaris --no-capture-output \\
        python isaac_lab_track/convert_urdf.py
"""

import argparse
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# AppLauncher must come before any isaaclab / isaacsim imports
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# ── Remaining imports (all after AppLauncher) ─────────────────────────────────
import omni.kit.commands
import omni.usd
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.extensions import enable_extension

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_SRC  = REPO_ROOT / "assets/iRonCub/robots/iRonCub-Mk1_1/model_stl.urdf"
STL_DIR   = REPO_ROOT / "assets/iRonCub/meshes/stl"
USD_OUT   = REPO_ROOT / "assets/iRonCub/robots/iRonCub-Mk1_1/ironcub.usd"


def _prepare_workdir(tmpdir: Path) -> tuple[Path, int]:
    """Create sanitized symlinks for STL files that contain hyphens in their names.

    USD SdfPaths cannot contain hyphens, so files like `sim_sea_2-5_*.stl`
    would produce invalid prim paths. We symlink them with underscores instead.

    Returns (workdir, n_symlinks).
    """
    n = 0
    for stl in STL_DIR.glob("*.stl"):
        safe = stl.name.replace("-", "_")
        link = tmpdir / safe
        link.symlink_to(stl.resolve())
        n += 1
    return tmpdir, n


def _preprocess_urdf(src: Path, tmpdir: Path, name_map: dict[str, Path]) -> Path:
    """Return path to a cleaned URDF written in tmpdir.

    Changes made:
    - Strips <gazebo> and top-level <sensor> elements (crash the v2 parser).
    - Strips <visual> elements (not needed; avoids redundant mesh load).
    - Rewrites collision <mesh filename=...> to absolute sanitized paths.
    """
    tree = ET.parse(src)
    root = tree.getroot()

    # 1. Remove Gazebo/sensor tags
    for el in [c for c in root if c.tag in ("gazebo", "sensor")]:
        root.remove(el)

    # 2. Strip visuals; rewrite collision mesh paths
    for link in root.findall("link"):
        for vis in link.findall("visual"):
            link.remove(vis)
        for col in link.findall("collision"):
            for geom in col.findall("geometry"):
                for mesh in geom.findall("mesh"):
                    orig_name = Path(mesh.get("filename", "")).name
                    if orig_name in name_map:
                        mesh.set("filename", str(name_map[orig_name]))

    out = tmpdir / "ironcub_clean.urdf"
    tree.write(out, encoding="unicode", xml_declaration=True)
    return out


def main() -> None:
    if not URDF_SRC.exists():
        print(f"ERROR: URDF not found: {URDF_SRC}")
        sys.exit(1)

    print(f"\nConverting URDF → USD")
    print(f"  Input : {URDF_SRC}")
    print(f"  Output: {USD_OUT}\n")

    tmpdir = Path(tempfile.mkdtemp())
    try:
        # Build symlink map  orig_name → safe_symlink_path
        _, n = _prepare_workdir(tmpdir)
        print(f"  Created {n} sanitized symlinks in {tmpdir}")
        name_map = {stl.name: tmpdir / stl.name.replace("-", "_")
                    for stl in STL_DIR.glob("*.stl")}

        clean_urdf = _preprocess_urdf(URDF_SRC, tmpdir, name_map)
        print(f"  Preprocessed URDF: {clean_urdf}")

        # Enable URDF importer extension
        enable_extension("isaacsim.asset.importer.urdf")

        # Fresh stage
        stage_utils.create_new_stage()

        # Import
        _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
        cfg.set_merge_fixed_joints(True)
        cfg.set_fix_base(False)          # iRonCub is free-floating
        cfg.set_make_default_prim(True)
        cfg.set_create_physics_scene(False)
        cfg.set_distance_scale(1.0)

        result, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(clean_urdf),
            import_config=cfg,
        )
        if not result:
            print("ERROR: URDFParseAndImportFile returned False — see Omniverse log.")
            sys.exit(1)

        print(f"  Imported prim: {prim_path}")

        # Export USD
        USD_OUT.parent.mkdir(parents=True, exist_ok=True)
        omni.usd.get_context().get_stage().Export(str(USD_OUT))
        size_mb = USD_OUT.stat().st_size / 1e6
        print(f"\nConversion complete → {USD_OUT}  ({size_mb:.1f} MB)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
