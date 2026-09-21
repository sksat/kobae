// kobae — whole-CNS LIF kernels (WGSL, wgpu)
//
// One "batch" = DELAY (18) integration steps of dt = 0.1 ms. A spike emitted at
// batch k / substep s is delivered at batch k+1 / substep s, so every arrival a
// batch needs is already accumulated in gin[s][target] before the batch starts.
//
// Per batch, three dispatches:
//   integrate : one thread per neuron, loops the 18 substeps, appends spikes to
//               the delivery buffer (capacity n: refractory 22 > delay 18 means a
//               neuron spikes at most once per batch) and to the observation ring
//   prep      : one thread, turns the delivery count into indirect dispatch args
//   scatter   : one workgroup per spike, strides the CSR row with int atomics
//
// Semantics follow DOOMFLY doom/engine.py exactly (see cpu.py):
//   if refractory>0: refractory-=1
//   if refractory==0: v = rest + (v-rest)*av + drive*(1-av) + g*coupling; g*=ag; spike if v>thr
//   arrivals from t-18 added to g unless refractory>0 (post-decrement)
//   spikers of this step: v=rest, g=0, refractory=RFC
//
// Fixed point: synaptic input is accumulated as i32 in units of 1/G_SCALE mV
// (10240/mV: 0.275 mV = 2816 exactly). Integer atomics are order-independent,
// so the arrival sums are bit-reproducible on one device.

struct Params {
  n: u32,
  delay: u32,
  ring_cap: u32,       // observation ring capacity (entries)
  _p0: u32,
  av: f32,
  ag: f32,
  coupling: f32,
  rest: f32,
  thr: f32,
  rfc: f32,
  inv_scale: f32,
  graded_gain: f32,    // fixed-point units delivered per batch per unit rate: G_SCALE * f_max * batch_s
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> ptr: array<u32>;
@group(0) @binding(2) var<storage, read> post: array<u32>;
@group(0) @binding(3) var<storage, read> wq: array<i32>;        // fixed-point weights
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> g: array<f32>;
@group(0) @binding(6) var<storage, read_write> refr: array<i32>;
@group(0) @binding(7) var<storage, read> drive: array<f32>;
@group(0) @binding(8) var<storage, read_write> gin: array<atomic<i32>>;   // [delay][n]
@group(0) @binding(9) var<storage, read_write> deliver: array<u32>;      // [n] this batch: (neuron<<5)|substep
// ctl: [0]=deliver count, [4]=ring head (monotonic), [5]=ring overflow flag, [6]=batch index
@group(0) @binding(10) var<storage, read_write> ctl: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> counts: array<u32>;
// observation ring: (batch, packed) pairs; written by integrate, read + rewound by the host
@group(0) @binding(12) var<storage, read_write> ring: array<vec2<u32>>;
// stage 2 (graded optic lobe): kind[i] = 1 for graded cells (no spikes; continuous output rate)
@group(0) @binding(13) var<storage, read> kind: array<u32>;
@group(0) @binding(14) var<storage, read_write> rate: array<f32>;     // [n] graded output in [0,1]
// CSC restricted to edges whose presynaptic cell is graded: gptr[n+1], gpre, gwq
@group(0) @binding(15) var<storage, read> gptr: array<u32>;
@group(0) @binding(16) var<storage, read> gpre: array<u32>;
@group(0) @binding(17) var<storage, read> gwq: array<i32>;
// indirect dispatch args for scatter; bound in its own group so it is not a storage
// binding of the scatter dispatch (wgpu forbids STORAGE_READ_WRITE + INDIRECT in one scope)
@group(1) @binding(0) var<storage, read_write> indirect: array<u32>;

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  var vi = v[i];
  var gi = g[i];
  var r = refr[i];
  let d = drive[i];
  let batch = atomicLoad(&ctl[6]);
  let graded = kind[i] == 1u;
  var c = 0u;
  for (var s = 0u; s < P.delay; s = s + 1u) {
    if (r > 0) { r = r - 1; }
    var spiked = false;
    if (r == 0) {
      vi = P.rest + (vi - P.rest) * P.av + d * (1.0 - P.av) + gi * P.coupling;
      gi = gi * P.ag;
      if (vi > P.thr && !graded) {
        spiked = true;
        c = c + 1u;
        let packed = (i << 5u) | s;
        let k = atomicAdd(&ctl[0], 1u);
        if (k < P.n) { deliver[k] = packed; }      // k < n by design (one spike per neuron per batch); guard anyway
        let h = atomicAdd(&ctl[4], 1u);
        if (h < P.ring_cap) { ring[h] = vec2<u32>(batch, packed); } else { atomicStore(&ctl[5], 1u); }
      }
    }
    let a = atomicExchange(&gin[s * P.n + i], 0);
    if (r == 0) { gi = gi + f32(a) * P.inv_scale; }
    if (spiked) { vi = P.rest; gi = 0.0; r = i32(P.rfc); }
  }
  if (graded) {
    // graded cells: clamp the membrane so a strongly driven cell saturates instead of running away
    vi = min(vi, P.thr + (P.thr - P.rest));
    rate[i] = clamp((vi - P.rest) / (P.thr - P.rest), 0.0, 1.0);
  }
  v[i] = vi;
  g[i] = gi;
  refr[i] = r;
  counts[i] = counts[i] + c;
}

// stage 2: pull the graded inputs of every cell (fixed order -> deterministic) and lump one batch's
// worth of continuous transmission into substep 0 of the next batch
@compute @workgroup_size(64)
fn graded_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= P.n) { return; }
  let e1 = gptr[j + 1u];
  var acc = 0.0;
  for (var e = gptr[j]; e < e1; e = e + 1u) {
    acc = acc + f32(gwq[e]) * rate[gpre[e]];
  }
  if (acc != 0.0) {
    atomicAdd(&gin[j], i32(round(acc * P.graded_gain)));
  }
}

@compute @workgroup_size(1)
fn prep() {
  // deliver[] has capacity n; a corrupted/overflowed count must never dispatch more than that
  let cnt = min(atomicLoad(&ctl[0]), P.n);
  atomicStore(&ctl[0], cnt);
  indirect[0] = min(cnt, 65535u);
  indirect[1] = (cnt + 65534u) / 65535u;
  indirect[2] = 1u;
}

@compute @workgroup_size(64)
fn scatter(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let k = wg.y * 65535u + wg.x;
  let cnt = atomicLoad(&ctl[0]);
  if (k >= cnt) { return; }
  let packed = deliver[k];
  let i = packed >> 5u;
  let base = (packed & 31u) * P.n;
  let e1 = ptr[i + 1u];
  for (var e = ptr[i] + lid.x; e < e1; e = e + 64u) {
    atomicAdd(&gin[base + post[e]], wq[e]);
  }
}

// reset the delivery count after scatter (single thread)
@compute @workgroup_size(1)
fn finish() {
  atomicStore(&ctl[0], 0u);
  atomicAdd(&ctl[6], 1u);
}
