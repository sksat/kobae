"""Build the retained MaleCNS v1.0 graph as CSR + stimulus/readout index sets.

Node and edge policy follow DOOMFLY (MIT, github.com/nftechie/doomfly,
doom/connectome.py + doom/transmitters.py + doom/prepare.py) so that spike
trains can be compared 1:1 with its reference kernel:

* nodes  : every annotation row with a non-empty superclass and status != Glia,
           sorted by bodyId; node index = position in that sorted list
* edges  : every released edge whose pre and post are retained nodes; no
           synapse-count threshold, self edges kept; CSR rows in stable
           pre-sorted file order
* weight : synapse_count * sign(pre neurotransmitter) * 0.275 mV, sign +1 for
           acetylcholine, -1 for GABA/glutamate/histamine, +1 when ambiguous
* retina : R1-R6 photoreceptors mapped to the modal hex column of their
           L1/L2/L3 targets, embedded into a [0,1]^2 uv "screen" (left eye
           on the left 60 %, right eye on the right 60 %, overlapping)
* lamina : L1/L2/L3/L5 ; sugar : LB3c ; readouts : DNa02 DNp09 MDN MN9 DNp20 DNpe017

Extras beyond DOOMFLY: soma position per neuron (for the viewer), cell type,
soma side, superclass and neurotransmitter strings, ORN groups by glomerulus.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "edges": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
SHA256 = {
    "annotations": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    "neurotransmitters": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    "edges": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
}
CONTACT_GAIN_MV = 0.275
READOUT_TYPES = ["DNa02", "DNp09", "MDN", "MN9", "DNp20", "DNpe017"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def exact_ids(values) -> np.ndarray:
    items = np.asarray(values)
    if items.dtype.kind == "f":
        raise ValueError("Neuron IDs must be integers, never floats.")
    if items.dtype.kind in "iu":
        if np.any(items < 0):
            raise ValueError("Neuron IDs cannot be negative.")
        return items.astype(np.uint64)
    return np.asarray([str(v) for v in items], dtype=np.uint64)


def transmitter_signs(transmitters, ambiguous_sign: int = 1):
    """DOOMFLY's coarse fast-transmission sign proxy (not receptor physiology)."""
    signs, uncertain = [], []
    for value in transmitters:
        tokens = set(str(value).lower().split(","))
        fast = ({1} if "acetylcholine" in tokens else set()) | (
            {-1} if tokens & {"gaba", "glutamate", "histamine"} else set())
        ambiguous = len(fast) != 1
        signs.append(ambiguous_sign if ambiguous else next(iter(fast)))
        uncertain.append(ambiguous)
    return np.asarray(signs, dtype=np.int8), np.asarray(uncertain, dtype=bool)


def _soma_xyz(a) -> np.ndarray:
    """Best-effort soma coordinates (nm or voxel units as released); NaN if absent."""
    n = len(a)
    xyz = np.full((n, 3), np.nan, dtype=np.float32)
    cols = set(a.columns)
    if {"somaLocation_x", "somaLocation_y", "somaLocation_z"} <= cols:
        for k, c in enumerate("xyz"):
            xyz[:, k] = a[f"somaLocation_{c}"].to_numpy(dtype=np.float64, na_value=np.nan)
        return xyz
    for name in ("somaLocation", "somaLocation_nm", "position", "location"):
        if name in cols:
            col = a[name]
            for i, v in enumerate(col.to_numpy(dtype=object)):
                if v is None:
                    continue
                try:
                    if isinstance(v, str):
                        v = json.loads(v) if v.startswith("[") else [float(t) for t in v.strip("()[] ").split(",")]
                    if len(v) == 3 and all(x is not None for x in v):
                        xyz[i] = [float(v[0]), float(v[1]), float(v[2])]
                except (ValueError, TypeError):
                    pass
            return xyz
    return xyz


def build(data_dir: Path, out: Path, verify: bool = True) -> dict:
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc

    t0 = time.time()
    paths = {k: data_dir / v for k, v in FILES.items()}
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(p)
        if verify:
            d = sha256(p)
            if d != SHA256[k]:
                raise ValueError(f"{p.name}: sha256 {d} != expected {SHA256[k]}")
    print(f"[graph] sources verified ({time.time() - t0:.0f}s)")

    a = feather.read_table(paths["annotations"]).to_pandas()
    nt = feather.read_table(paths["neurotransmitters"]).to_pandas().set_index("body")
    if not nt.index.is_unique:
        raise ValueError("Duplicate neurotransmitter body ids")
    source = exact_ids(a.bodyId)
    retain = a.superclass.notna() & a.superclass.astype(str).ne("")
    retain &= ~a.status.eq("Glia")
    order = np.argsort(source, kind="stable")
    keep = order[retain.to_numpy()[order]]
    nodes = a.iloc[keep].reset_index(drop=True)
    ids = source[keep]
    if np.any(ids[1:] <= ids[:-1]):
        raise ValueError("node ids not strictly increasing")
    n = len(nodes)
    predicted = nodes.bodyId.map(nt.consensus_nt)
    sign, uncertain = transmitter_signs(predicted)
    print(f"[graph] {n} retained neurons of {len(a)} rows; {int(uncertain.sum())} ambiguous-sign")

    # ---- edges: stream record batches, keep rows with both ends retained
    reader = ipc.open_file(pa.memory_map(str(paths["edges"]), "r"))
    pre_l, post_l, cnt_l = [], [], []
    src_rows = src_contacts = 0
    for b in range(reader.num_record_batches):
        batch = reader.get_batch(b)
        pre = exact_ids(batch.column("body_pre").to_numpy(zero_copy_only=False))
        post = exact_ids(batch.column("body_post").to_numpy(zero_copy_only=False))
        w = batch.column("weight").to_numpy(zero_copy_only=False)
        if np.any(w < 1) or np.any(w != np.floor(w)):
            raise ValueError("synapse counts must be positive integers")
        i, j = np.searchsorted(ids, pre), np.searchsorted(ids, post)
        ok = (i < n) & (j < n)
        ok &= ids[np.minimum(i, n - 1)] == pre
        ok &= ids[np.minimum(j, n - 1)] == post
        src_rows += len(pre)
        src_contacts += int(w.sum(dtype=np.uint64))
        pre_l.append(i[ok].astype(np.uint32))
        post_l.append(j[ok].astype(np.uint32))
        cnt_l.append(w[ok].astype(np.uint32))
    pre = np.concatenate(pre_l); post = np.concatenate(post_l); count = np.concatenate(cnt_l)
    del pre_l, post_l, cnt_l
    order = np.argsort(pre, kind="stable")
    ptr = np.r_[0, np.cumsum(np.bincount(pre, minlength=n))].astype(np.int64)
    post = post[order].astype(np.int32)
    count = count[order]
    weight = (count.astype(np.float32) * sign[pre[order]] * CONTACT_GAIN_MV).astype(np.float32)
    E = len(post)
    print(f"[graph] {E} retained edges of {src_rows} rows, {int(count.sum(dtype=np.uint64))} contacts "
          f"of {src_contacts}; self edges {int(np.count_nonzero(pre[order] == post))} ({time.time() - t0:.0f}s)")

    # ---- retina projection (DOOMFLY prepare.py)
    a_idx = a.set_index("bodyId").loc[nodes.bodyId]
    receptor = a_idx.type.eq("R1-R6").to_numpy()
    anchors = a_idx.type.isin(["L1", "L2", "L3"]).to_numpy() & a_idx.assignedOlHex1.notna().to_numpy()
    pre_s = pre[order]
    selected = receptor[pre_s] & anchors[post]
    hex1 = a_idx.assignedOlHex1.to_numpy(dtype=np.float64, na_value=np.nan)
    hex2 = a_idx.assignedOlHex2.to_numpy(dtype=np.float64, na_value=np.nan)
    cols: dict[int, dict] = {}
    for i, j, w in zip(pre_s[selected], post[selected], count[selected]):
        key = (float(hex1[j]), float(hex2[j]))
        d = cols.setdefault(int(i), {})
        d[key] = d.get(key, 0) + int(w)
    indices, xy, confidence, hexes = [], [], [], []
    for i, c in sorted(cols.items()):
        h = max(c, key=c.get); total = sum(c.values())
        indices.append(i); hexes.append(h); confidence.append(c[h] / total)
        xy.append((h[0] - .5 * h[1], np.sqrt(3) / 2 * h[1]))
    xy = np.asarray(xy); indices = np.asarray(indices, dtype=np.int32)
    uv = np.empty_like(xy)
    sides = a_idx.rootSide.to_numpy()[indices]
    for side in ["L", "R"]:
        m = sides == side; z = xy[m]; z = (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0))
        uv[m, 0] = (.60 * z[:, 0] if side == "L" else .40 + .60 * (1 - z[:, 0]))
        uv[m, 1] = 1 - z[:, 1]
    ctype = nodes.type.fillna("").astype(str).to_numpy()
    lamina = np.flatnonzero(np.isin(ctype, ["L1", "L2", "L3", "L5"])).astype(np.int32)
    sugar = np.flatnonzero(ctype == "LB3c").astype(np.int32)
    soma_side = a_idx.somaSide.fillna("").astype(str).to_numpy() if "somaSide" in a_idx.columns else np.full(n, "", dtype=object)
    readouts = [{"index": int(i), "id": str(nodes.bodyId.iloc[i]), "type": ctype[i], "side": str(soma_side[i])}
                for i in np.flatnonzero(np.isin(ctype, READOUT_TYPES))]
    print(f"[graph] retina mapped {len(indices)}/{int(receptor.sum())}, lamina {len(lamina)}, sugar {len(sugar)}, "
          f"readouts {len(readouts)}")

    soma = _soma_xyz(a_idx)
    tosoma = np.full((n, 3), np.nan, dtype=np.float32)
    if "tosomaLocation" in a_idx.columns:
        for i, v in enumerate(a_idx.tosomaLocation.to_numpy(dtype=object)):
            if v is not None and len(v) == 3:
                tosoma[i] = [float(v[0]), float(v[1]), float(v[2])]
    print(f"[graph] soma positions for {int(np.isfinite(soma[:, 0]).sum())} neurons "
          f"(+{int(np.isfinite(tosoma[:, 0]).sum())} tosoma)")
    receptor_type = a_idx.receptorType.fillna("").astype(str).to_numpy() if "receptorType" in a_idx.columns else np.full(n, "", dtype=object)
    fru_dsx = a_idx.fruDsx.fillna("").astype(str).to_numpy() if "fruDsx" in a_idx.columns else np.full(n, "", dtype=object)

    superclass = nodes.superclass.fillna("unassigned").astype(str).to_numpy()
    orn = np.flatnonzero(np.char.startswith(ctype.astype(str), "ORN_")).astype(np.int32)
    meta = {
        "dataset": "malecns_v1", "neurons": n, "edges": E,
        "synaptic_contacts": int(count.sum(dtype=np.uint64)),
        "source_edge_rows": src_rows, "source_contacts": src_contacts,
        "retina_total": int(receptor.sum()), "retina_mapped": len(indices),
        "projection_confidence_median": float(np.median(confidence)),
        "uncertain_sign_neurons": int(uncertain.sum()),
        "readouts": readouts, "source_sha256": SHA256, "contact_gain_mv": CONTACT_GAIN_MV,
        "superclass_counts": {str(k): int(v) for k, v in zip(*np.unique(superclass, return_counts=True))},
        "orn_neurons": int(len(orn)),
        "policy": "DOOMFLY: superclass assigned & not Glia; all released edges between retained nodes; "
                  "weight = count*sign*0.275 mV; retina R1-R6 modal L1/L2/L3 hex column; lamina L1/L2/L3/L5; sugar LB3c",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ptr=ptr, post=post, weight=weight, count=count,
             ids=ids.astype(np.int64), retina=indices, uv=uv.astype(np.float32),
             confidence=np.asarray(confidence, dtype=np.float32), lamina=lamina, sugar=sugar, orn=orn,
             superclass=superclass.astype("U64"), cell_type=ctype.astype("U64"),
             soma_side=np.asarray(soma_side, dtype="U8"),
             neurotransmitter=predicted.fillna("").astype(str).to_numpy().astype("U64"),
             sign=sign, soma=soma, tosoma=tosoma,
             receptor_type=np.asarray(receptor_type, dtype="U32"), fru_dsx=np.asarray(fru_dsx, dtype="U32"))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[graph] wrote {out} ({out.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s")
    return meta


class Graph:
    """Loaded graph arrays (read-only views)."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        for k in z.files:
            setattr(self, k, z[k])
        self.n = len(self.ids)
        self.E = len(self.post)
        self.meta = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
        if self.ptr.shape != (self.n + 1,) or self.ptr[-1] != self.E:
            raise ValueError("invalid CSR")

    def group(self, kind: str) -> np.ndarray:
        if kind == "sugar":
            return self.sugar
        if kind == "retina":
            return self.retina
        if kind == "lamina":
            return self.lamina
        if kind == "orn":
            return self.orn
        raise KeyError(kind)
