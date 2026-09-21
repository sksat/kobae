// Flight view: the flybody rig and the arena rendered in three.js, driven by streamed poses.
// MuJoCo frame: z up, centimetres. Quaternions arrive as (w, x, y, z).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export async function createFlightView(canvas) {
  const rig = await (await fetch("/rig/rig.json")).json();
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(1);
  renderer.shadowMap.enabled = true;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9ec6ea);
  scene.fog = new THREE.Fog(0x9ec6ea, 120, 420);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.02, 1500);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.dampingFactor = 0.1;
  controls.minDistance = 0.3; controls.maxDistance = 200;

  scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x3a4a30, 0.9));
  const sun = new THREE.DirectionalLight(0xfff2dd, 1.4);
  sun.position.set(40, 25, 90); sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.left = -30; sun.shadow.camera.right = 30; sun.shadow.camera.top = 30; sun.shadow.camera.bottom = -30;
  sun.shadow.camera.near = 1; sun.shadow.camera.far = 250;
  scene.add(sun); scene.add(sun.target);

  // ---- arena from the exported geoms (everything not on a walker body)
  const walkerBodies = new Set(rig.walker_bodies);
  const materials = new Map();
  const matFor = (c) => {
    const k = c.join(",");
    if (!materials.has(k)) materials.set(k, new THREE.MeshStandardMaterial({ color: new THREE.Color(c[0], c[1], c[2]), transparent: c[3] < 1, opacity: c[3], roughness: 0.75, metalness: 0.0, side: c[3] < 1 ? THREE.DoubleSide : THREE.FrontSide }));
    return materials.get(k);
  };
  const geoms = rig.meshes.map((m) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(m.v, 3));
    g.setIndex(m.f);
    g.computeVertexNormals();
    return g;
  });
  const quatOf = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);

  // checker floor
  const floorTex = (() => {
    const c = document.createElement("canvas"); c.width = c.height = 256; const x = c.getContext("2d");
    x.fillStyle = "#6c8264"; x.fillRect(0, 0, 256, 256); x.fillStyle = "#3d5040"; x.fillRect(0, 0, 128, 128); x.fillRect(128, 128, 128, 128);
    const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(100, 100); t.anisotropy = 4; return t;
  })();
  for (const g of rig.geoms) {
    if (walkerBodies.has(g.body)) continue;
    let mesh;
    if (g.type === 0) {
      mesh = new THREE.Mesh(new THREE.PlaneGeometry(g.size[0] * 2, g.size[1] * 2), new THREE.MeshStandardMaterial({ map: floorTex, roughness: 1 }));
      mesh.receiveShadow = true;
    } else if (g.type === 5) {
      mesh = new THREE.Mesh(new THREE.CylinderGeometry(g.size[0], g.size[0], g.size[1] * 2, 24), matFor(g.color));
      mesh.rotation.x = Math.PI / 2;   // three cylinders are y-up; MuJoCo's are z-up
      mesh.castShadow = true;
    } else if (g.type === 6) {
      mesh = new THREE.Mesh(new THREE.BoxGeometry(g.size[0] * 2, g.size[1] * 2, g.size[2] * 2), matFor(g.color));
      mesh.castShadow = true;
    } else continue;
    const holder = new THREE.Group();
    holder.position.set(g.pos[0], g.pos[1], g.pos[2]); holder.quaternion.copy(quatOf(g.quat));
    holder.add(mesh); scene.add(holder);
  }

  // ---- fly rig: one Group per body, meshes attached with their local pose
  const bodyGroups = new Map();
  for (const id of rig.walker_bodies) { const grp = new THREE.Group(); bodyGroups.set(id, grp); scene.add(grp); }
  for (const g of rig.geoms) {
    if (!walkerBodies.has(g.body) || g.type !== 7) continue;
    const mesh = new THREE.Mesh(geoms[g.mesh], matFor(g.color));
    mesh.position.set(g.pos[0], g.pos[1], g.pos[2]); mesh.quaternion.copy(quatOf(g.quat));
    mesh.castShadow = g.color[3] >= 1; mesh.receiveShadow = false;
    bodyGroups.get(g.body).add(mesh);
  }
  const names = new Map(rig.bodies.map(b => [b.id, b.name]));
  let thoraxId = rig.bodies.find(b => b.name === "walker/thorax")?.id ?? rig.walker_bodies[0];

  // camera: orbit around the thorax; keep the user's offset when the fly moves
  const target = new THREE.Vector3(0, 0, 1);
  camera.position.set(-1.1, -1.7, 0.9); controls.target.copy(target);
  const prevTarget = target.clone();
  let hasPose = false;

  function applyPoses(ids, poses) {
    for (let k = 0; k < ids.length; k++) {
      const grp = bodyGroups.get(ids[k]); if (!grp) continue;
      const o = k * 7;
      grp.position.set(poses[o], poses[o + 1], poses[o + 2]);
      grp.quaternion.set(poses[o + 4], poses[o + 5], poses[o + 6], poses[o + 3]);
    }
    const th = bodyGroups.get(thoraxId);
    if (th) {
      prevTarget.copy(controls.target);
      controls.target.copy(th.position);
      camera.position.add(controls.target).sub(prevTarget);   // follow: shift the camera by the same delta
      sun.position.copy(th.position).add(new THREE.Vector3(40, 25, 90)); sun.target.position.copy(th.position);
      hasPose = true;
    }
  }

  function resize() {
    const box = canvas.parentElement.getBoundingClientRect();
    const w = Math.max(1, Math.round(box.width)), h = Math.max(1, Math.round(box.height));
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  function render() { controls.update(); renderer.render(scene, camera); }
  const views = {
    chase: () => { const t = controls.target; camera.position.set(t.x - 1.1, t.y - 1.7, t.z + 0.9); },
    top: () => { const t = controls.target; camera.position.set(t.x, t.y - 0.01, t.z + 4); },
    side: () => { const t = controls.target; camera.position.set(t.x + 2.5, t.y, t.z + 0.4); },
    wide: () => { const t = controls.target; camera.position.set(t.x - 15, t.y - 25, t.z + 15); },
  };
  return { applyPoses, resize, render, views, get hasPose() { return hasPose; }, tris: rig.meshes.reduce((a, m) => a + m.f.length / 3, 0) };
}
