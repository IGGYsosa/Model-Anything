import * as THREE from 'three';
import { SplatEdit, SplatEditSdf, SplatEditSdfType, SplatMesh } from '@sparkjsdev/spark';
import type { BuiltRoom, RoomLayout, RoomModule, V3 } from './types';
import { RESCAN_COLORS } from './splat-room-colors';

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
  review?: SceneReview;
}

/** Where the scene is weak and where to rescan (apps/pipeline/review.py), room coordinates. */
export interface SceneReview {
  weakPoints: string;
  weakPointCount: number;
  weakFraction: number;
  areas: RescanArea[];
  views: RescanView[];
}
export interface RescanArea {
  id: number;
  /** inaccurate: the result differs from a held-out photo; few-photos: too few photos see it */
  reason: 'inaccurate' | 'few-photos';
  severity: number;
  points: number;
  center: V3; min: V3; max: V3; normal: V3;
  side: string;
  viewpoint: { position: V3; target: V3 };
  /** index into views of a held-out photo that sees this area */
  view: number | null;
  text: string;
}
export interface RescanView {
  name: string;
  psnr: number;
  /** URLs (resolved against scene.json) of the real photo and the render from the same pose */
  photo: string;
  render: string;
}
/** The weak-spot markers drawn in the scene */
export interface RescanOverlay {
  review: SceneReview;
  group: THREE.Group;
  highlight(id: number | null): void;
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

function label(text: string, color: string) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(32, 32, 28, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#1a1a1a';
  ctx.font = 'bold 30px system-ui, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 34);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
  sprite.scale.setScalar(0.16);
  sprite.renderOrder = 1002;
  return sprite;
}

/** Boxes, numbered badges and dots for the review's weak areas. Drawn over the splat. */
async function rescanOverlay(base: URL, review: SceneReview): Promise<RescanOverlay> {
  const group = new THREE.Group();
  group.name = 'rescan-overlay';
  const response = await fetch(new URL(review.weakPoints, base)).catch(() => undefined);
  if (response?.ok) {
    const packed = new Float32Array(await response.arrayBuffer());
    const count = Math.floor(packed.length / 4);
    const positions = new Float32Array(count * 3), colors = new Float32Array(count * 3);
    const red = new THREE.Color(RESCAN_COLORS.inaccurate), amber = new THREE.Color(RESCAN_COLORS['few-photos']);
    for (let i = 0; i < count; i++) {
      positions.set(packed.subarray(i * 4, i * 4 + 3), i * 3);
      (packed[i * 4 + 3] > 0.5 ? red : amber).toArray(colors, i * 3);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    // small round dots of constant screen size: up close, world-sized points become big squares
    const disc = document.createElement('canvas');
    disc.width = disc.height = 32;
    const ctx = disc.getContext('2d')!;
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(16, 16, 14, 0, Math.PI * 2); ctx.fill();
    const dots = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 5, sizeAttenuation: false, map: new THREE.CanvasTexture(disc), alphaTest: 0.5,
      vertexColors: true, transparent: true, opacity: 0.75, depthWrite: false,
    }));
    dots.name = 'weak-points';
    dots.renderOrder = 1000;
    group.add(dots);
  }
  const markers = review.areas.map(area => {
    const color = RESCAN_COLORS[area.reason];
    const size = new THREE.Vector3(...area.max).sub(new THREE.Vector3(...area.min));
    const center = new THREE.Vector3(...area.max).add(new THREE.Vector3(...area.min)).multiplyScalar(0.5);
    const box = new THREE.BoxGeometry(size.x, size.y, size.z);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box),
      new THREE.LineBasicMaterial({ color, transparent: true, depthTest: false }));
    const fill = new THREE.Mesh(box, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.07, depthWrite: false }));
    const badge = label(String(area.id), color);
    badge.position.set(0, size.y / 2 + 0.16, 0);
    const marker = new THREE.Group();
    marker.name = `weak-area-${area.id}`;
    marker.position.copy(center);
    edges.renderOrder = fill.renderOrder = 1001;
    marker.add(fill, edges, badge);
    group.add(marker);
    return { id: area.id, edges, fill, badge };
  });
  // where to stand for the selected area's photos: a ring on the floor and a post to eye height
  const stand = new THREE.Group();
  stand.name = 'stand-here';
  const standColor = '#8fd18b';
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.24, 40),
    new THREE.MeshBasicMaterial({ color: standColor, transparent: true, depthTest: false, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.01;
  const post = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1.5, 0)]),
    new THREE.LineBasicMaterial({ color: standColor, transparent: true, depthTest: false }));
  ring.renderOrder = post.renderOrder = 1003;
  stand.add(ring, post);
  stand.visible = false;
  group.add(stand);
  return {
    review, group,
    highlight(id) {
      const area = review.areas.find(a => a.id === id);
      stand.visible = !!area;
      if (area) stand.position.set(area.viewpoint.position[0], 0, area.viewpoint.position[2]);
      for (const m of markers) {
        const on = id === null || m.id === id;
        (m.edges.material as THREE.LineBasicMaterial).opacity = on ? 1 : 0.25;
        (m.fill.material as THREE.MeshBasicMaterial).opacity = id === m.id ? 0.16 : on ? 0.07 : 0.03;
        (m.badge.material as THREE.SpriteMaterial).opacity = on ? 1 : 0.35;
        m.badge.scale.setScalar(id === m.id ? 0.22 : 0.16);
      }
    },
  };
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

      let rescan: RescanOverlay | undefined;
      if (scene.review?.areas) {
        rescan = await rescanOverlay(base, scene.review);
        for (const view of scene.review.views) {
          view.photo = new URL(view.photo, base).href;
          view.render = new URL(view.render, base).href;
        }
        group.add(rescan.group);
      }

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
        rescan,
      };
      return room;
    },
  };
}
