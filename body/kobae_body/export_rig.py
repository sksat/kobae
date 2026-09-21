"""Export the flybody rig (decimated meshes per body) and the arena for the browser viewer.

Output: rig.json  { bodies:[{id,name,parent}], geoms:[{body,mesh,color,pos,quat,type,size}], meshes:[{v:[..],f:[..]}] }
The meshes are decimated from flybody's 154 MB of OBJ (via the compiled MuJoCo model) to ~150k triangles total.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import fast_simplification

from .flight import FlightBody


def main(out: str, target_tris: int = 900):
    body = FlightBody("policies/flight.npz", "../data/flybody/wing_pattern_fmech.npy")
    m = body.physics.model.ptr
    name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""
    bodies = [{"id": i, "name": name(mujoco.mjtObj.mjOBJ_BODY, i), "parent": int(m.body_parentid[i])} for i in range(m.nbody)]
    meshes, mesh_index = [], {}
    geoms = []
    for g in range(m.ngeom):
        gname = name(mujoco.mjtObj.mjOBJ_GEOM, g)
        if gname.startswith("ghost") or m.geom_group[g] > 2:      # hide ghost and collision-only groups
            continue
        typ = int(m.geom_type[g])
        rgba = [round(float(x), 3) for x in m.geom_rgba[g]]
        if rgba[3] == 0:
            continue
        entry = {"body": int(m.body_parentid[m.geom_bodyid[g]]) if False else int(m.geom_bodyid[g]), "name": gname,
                 "color": rgba[:3], "pos": [round(float(x), 5) for x in m.geom_pos[g]],
                 "quat": [round(float(x), 5) for x in m.geom_quat[g]], "type": typ,
                 "size": [round(float(x), 5) for x in m.geom_size[g]]}
        if typ == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            if mid not in mesh_index:
                v0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                f0, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                v = m.mesh_vert[v0:v0 + nv].astype(np.float64); f = m.mesh_face[f0:f0 + nf].astype(np.int64)
                if nf > target_tris:
                    v, f = fast_simplification.simplify(v, f, target_count=target_tris)
                mesh_index[mid] = len(meshes)
                meshes.append({"v": np.round(v, 5).ravel().tolist(), "f": np.asarray(f).ravel().tolist()})
            entry["mesh"] = mesh_index[mid]
        geoms.append(entry)
    walker_bodies = {b["id"] for b in bodies if b["name"].startswith("walker/")}
    out_p = Path(out); out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"bodies": bodies, "geoms": geoms, "meshes": meshes, "walker_bodies": sorted(walker_bodies)}))
    tris = sum(len(x["f"]) // 3 for x in meshes)
    print(f"bodies {len(bodies)} geoms {len(geoms)} meshes {len(meshes)} tris {tris} -> {out_p} ({out_p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../src/kobae/viewer/rig/rig.json")
