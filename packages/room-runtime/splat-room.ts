import * as THREE from 'three';
import { SplatEdit, SplatEditSdf, SplatEditSdfType, SplatMesh } from '@sparkjsdev/spark';
import type { BuiltRoom, RoomLayout, RoomModule } from './types';

/** scene.json written by apps/pipeline/run_job.py: a trained splat in room coordinates
 * (meters, +Y up, floor at 0) plus the RoomPlan layout it lines up with. */
export interface ScenePackage {
  format: 'model-anything-scene/1';
  name: string;
  /** splat file (.sog, .spz, .ply), relative to scene.json */
  splat: string;
  layout: RoomLayout;
  camera: BuiltRoom['camera'];
  height?: number;
  metrics?: Record<string, number | string>;
  hints?: string[];
}

export async function loadScenePackage(url: string): Promise<ScenePackage> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`scene ${url}: HTTP ${response.status}`);
  const scene = await response.json().catch(() => { throw new Error(`scene ${url}: not JSON`); }) as ScenePackage;
  if (scene.format !== 'model-anything-scene/1' || !scene.splat || !scene.layout?.polygon?.length) {
    throw new Error(`scene ${url}: not a model-anything-scene/1 package`);
  }
  return scene;
}

function inside(polygon: number[][], x: number, z: number) {
  let hit = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [x1, z1] = polygon[i], [x2, z2] = polygon[j];
    if ((z1 > z) !== (z2 > z) && x < x1 + (z - z1) * (x2 - x1) / (z2 - z1)) hit = !hit;
  }
  return hit;
}

/** A photoreal room: the splat draws everything, the layout supplies the floor plan, the walk
 * boundary and a dollhouse cutaway (splats hidden by SDF edits instead of hiding wall meshes). */
export function splatRoom(id: string, sceneURL: string, scene: ScenePackage): RoomModule {
  return {
    id, label: scene.name,
    async build() {
      const base = new URL(sceneURL, location.href);
      const splat = new SplatMesh({ url: new URL(scene.splat, base).href });
      await splat.initialized;
      splat.name = 'photoreal-splat';
      const group = new THREE.Group();
      group.name = `scene-${id}`;
      group.add(splat);

      const { layout } = scene;
      const height = scene.height ?? 2.5;
      const up = new THREE.Vector3(0, 1, 0);
      // One edit per wall: a slab from just inside the wall plane to beyond its outer face.
      const walls = layout.walls.map(w => {
        const u = new THREE.Vector3(...w.u).setY(0).normalize();
        const n = new THREE.Vector3(-u.z, 0, u.x); // into the room (bundle_to_room.py: n = u × up)
        const mid = new THREE.Vector3((w.a[0] + w.b[0]) / 2, 0, (w.a[1] + w.b[1]) / 2);
        const box = new SplatEditSdf({ type: SplatEditSdfType.BOX, opacity: 0 });
        box.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(u, up, n));
        box.position.copy(mid).addScaledVector(n, -0.16).setY(height / 2);
        box.scale.set(w.length / 2 + 0.15, height / 2 + 0.3, 0.22);
        const edit = new SplatEdit({ name: 'wall-cutaway', sdfs: [box] });
        edit.visible = false;
        splat.add(edit);
        return { mid, n, edit };
      });
      // Above the walls: hide the ceiling when looking down into the room.
      const ceilingPlane = new SplatEditSdf({ type: SplatEditSdfType.PLANE, opacity: 0 });
      ceilingPlane.rotation.x = Math.PI / 2; // local +z (the kept side) points down
      ceilingPlane.position.y = height - 0.08;
      const ceiling = new SplatEdit({ name: 'ceiling-cutaway', sdfs: [ceilingPlane] });
      ceiling.visible = false;
      splat.add(ceiling);

      const xs = layout.polygon.map(p => p[0]), zs = layout.polygon.map(p => p[1]);
      const bounds = new THREE.Box3(new THREE.Vector3(Math.min(...xs), 0, Math.min(...zs)),
        new THREE.Vector3(Math.max(...xs), height, Math.max(...zs)));
      const center = bounds.getCenter(new THREE.Vector3());
      const toEye = new THREE.Vector3();
      const room: BuiltRoom = {
        kind: 'splat',
        layout,
        group,
        lights: new THREE.Group(),
        walls: [],
        ceiling: [],
        wallTop: height,
        bounds,
        walkable: (x, z) => inside(layout.polygon, x, z),
        camera: scene.camera,
        environment: { center: [center.x, 1.2, center.z], intensity: 0 },
        cutaway(eye, walking) {
          for (const w of walls) w.edit.visible = !walking && toEye.copy(eye).sub(w.mid).dot(w.n) < 0;
          ceiling.visible = !walking && eye.y >= height;
        },
        loadReference: () => Promise.reject(new Error('A photoreal scene has no separate reference scan.')),
        referenceAvailable: false,
        referenceKind: 'scan',
        photos: [],
      };
      return room;
    },
  };
}
