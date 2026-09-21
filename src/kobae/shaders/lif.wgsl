// kobae — whole-CNS LIF kernels (WGSL, wgpu)
//
// One "batch" = DELAY (18) integration steps of dt = 0.1 ms. A spike emitted at
// batch k / substep s is delivered at batch k+1 / substep s, so every arrival a
// batch needs is already accumulated in gin[s][target] before the batch starts.
//
// Per batch, three dispatches:
//   integrate : one thread per neuron, loops the 18 substeps, appends spikes
//   prep      : one thread, turns the spike-log head into indirect dispatch args
//   scatter   : one workgroup per spike, strides the CSR row with int atomics
//
// Semantics follow DOOMFLY doom/engine.py exactly (see cpu.py):
//   if refractory>0: refractory-=1
//   if refractory==0: v = rest + (v-rest)*av + drive*(1-av) + g*coupling; g*=ag; spike if v>thr
//   arrivals from t-18 added to g unless refractory>0 (post-decrement)
//   spikers of this step: v=rest, g=0, refractory=RFC

struct Params {
  n: u32,
  delay: u32,
  log_cap: u32,
  _p0: u32,
  av: f32,
  ag: f32,
  coupling: f32,
  rest: f32,
  thr: f32,
  rfc: f32,
  inv_scale: f32,
  _p1: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> ptr: array<u32>;
@group(0) @binding(2) var<storage, read> post: array<u32>;
@group(0) @binding(3) var<storage, read> wq: array<i32>;        // fixed-point weights (mV * G_SCALE)
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> g: array<f32>;
@group(0) @binding(6) var<storage, read_write> refr: array<i32>;
@group(0) @binding(7) var<storage, read> drive: array<f32>;
@group(0) @binding(8) var<storage, read_write> gin: array<atomic<i32>>;   // [delay][n]
@group(0) @binding(9) var<storage, read_write> slog: array<u32>;         // spike log: (neuron<<5)|substep
// meta: [0]=log head, [1]=batch start, [2]=count, [3]=offset, [4..6]=indirect dispatch args
@group(0) @binding(10) var<storage, read_write> meta: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> counts: array<u32>;

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  var vi = v[i];
  var gi = g[i];
  var r = refr[i];
  let d = drive[i];
  var c = 0u;
  for (var s = 0u; s < P.delay; s = s + 1u) {
    if (r > 0) { r = r - 1; }
    var spiked = false;
    if (r == 0) {
      vi = P.rest + (vi - P.rest) * P.av + d * (1.0 - P.av) + gi * P.coupling;
      gi = gi * P.ag;
      if (vi > P.thr) {
        spiked = true;
        c = c + 1u;
        let idx = atomicAdd(&meta[0], 1u);
        if (idx < P.log_cap) { slog[idx] = (i << 5u) | s; }
      }
    }
    let a = atomicExchange(&gin[s * P.n + i], 0);
    if (r == 0) { gi = gi + f32(a) * P.inv_scale; }
    if (spiked) { vi = P.rest; gi = 0.0; r = i32(P.rfc); }
  }
  v[i] = vi;
  g[i] = gi;
  refr[i] = r;
  counts[i] = counts[i] + c;
}

@compute @workgroup_size(1)
fn prep() {
  let head = atomicLoad(&meta[0]);
  let start = atomicLoad(&meta[1]);
  let cnt = head - start;
  atomicStore(&meta[2], cnt);
  atomicStore(&meta[3], start);
  atomicStore(&meta[4], min(cnt, 65535u));
  atomicStore(&meta[5], (cnt + 65534u) / 65535u);
  atomicStore(&meta[6], 1u);
  atomicStore(&meta[1], head);
}

@compute @workgroup_size(64)
fn scatter(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let k = wg.y * 65535u + wg.x;
  let cnt = atomicLoad(&meta[2]);
  if (k >= cnt) { return; }
  let idx = atomicLoad(&meta[3]) + k;
  if (idx >= P.log_cap) { return; }
  let packed = slog[idx];
  let i = packed >> 5u;
  let base = (packed & 31u) * P.n;
  let e1 = ptr[i + 1u];
  for (var e = ptr[i] + lid.x; e < e1; e = e + 64u) {
    atomicAdd(&gin[base + post[e]], wq[e]);
  }
}
