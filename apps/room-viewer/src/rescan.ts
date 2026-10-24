import type { RescanArea, RescanOverlay } from '../../../packages/room-runtime/splat-room';
import { RESCAN_COLORS } from '../../../packages/room-runtime/splat-room-colors';
import type { V3 } from '../../../packages/room-runtime/types';

const REASON = { inaccurate: 'Blurry or smeared', 'few-photos': 'Too few photos' } as const;

/** A camera that shows the whole area with some surroundings, from the side new photos should be
 * taken from (the green marker shows where to stand). */
function frame(area: RescanArea): [V3, V3] {
  const [tx, ty, tz] = area.viewpoint.target, [px, py, pz] = area.viewpoint.position;
  const dir = [px - tx, py - ty, pz - tz];
  const length = Math.hypot(...dir) || 1;
  const extent = Math.max(...area.max.map((v, i) => v - area.min[i]));
  const distance = Math.max(2.2, extent * 1.8 / (2 * Math.tan(Math.PI * 55 / 360)));
  return [[tx + dir[0] / length * distance, Math.max(py, ty) + 0.6, tz + dir[2] / length * distance], [tx, ty, tz]];
}

/** The Rescan sidebar section: weak areas of a photoreal scene, where to stand, and the real
 * photo beside the result. Hidden for rooms without a review. */
export function showRescan(host: HTMLElement, rescan: RescanOverlay | undefined,
  actions: { focus(position: V3, target: V3): void; redraw(): void }) {
  host.hidden = !rescan;
  const list = host.querySelector<HTMLElement>('.rescan-list')!;
  const detail = host.querySelector<HTMLElement>('.rescan-detail')!;
  const summary = host.querySelector<HTMLElement>('#rescan-summary')!;
  const show = host.querySelector<HTMLInputElement>('#rescan-show')!;
  list.replaceChildren();
  detail.hidden = true;
  if (!rescan) return;
  const { review } = rescan;
  summary.textContent = review.areas.length
    ? `${review.areas.length} spot${review.areas.length === 1 ? '' : 's'} · ${Math.round(review.weakFraction * 100)}% of surfaces`
    : 'Looks complete';
  show.checked = rescan.group.visible = true;
  show.onchange = () => { rescan.group.visible = show.checked; actions.redraw(); };

  let selected: number | null = null;
  const select = (area: RescanArea | null) => {
    selected = area?.id ?? null;
    rescan.highlight(selected);
    for (const button of list.querySelectorAll<HTMLButtonElement>('button')) {
      button.setAttribute('aria-pressed', String(Number(button.dataset.area) === selected));
    }
    if (!area) { detail.hidden = true; actions.redraw(); return; }
    if (!show.checked) { show.checked = rescan.group.visible = true; }
    actions.focus(...frame(area));
    const view = area.view === null ? undefined : review.views[area.view];
    detail.replaceChildren();
    const text = document.createElement('p');
    text.textContent = area.text;
    detail.append(text);
    if (view) {
      const figures = document.createElement('div');
      figures.className = 'rescan-compare';
      for (const [src, caption] of [[view.photo, 'Your photo'], [view.render, 'Photoreal result']]) {
        const figure = document.createElement('figure');
        const img = document.createElement('img');
        img.src = src; img.alt = `${caption}, area ${area.id}`; img.loading = 'lazy';
        const cap = document.createElement('figcaption');
        cap.textContent = caption;
        figure.append(img, cap);
        figures.append(figure);
      }
      detail.append(figures);
    } else {
      const note = document.createElement('p');
      note.className = 'help';
      note.textContent = 'No held-out photo sees this spot, so there is nothing to compare. The camera now stands where new photos should be taken.';
      detail.append(note);
    }
    detail.hidden = false;
  };

  for (const area of review.areas) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.area = String(area.id);
    button.setAttribute('aria-pressed', 'false');
    button.style.setProperty('--marker', RESCAN_COLORS[area.reason]);
    button.innerHTML = '<span class="rescan-badge"></span><span class="rescan-label"><strong></strong><small></small></span>';
    button.querySelector('.rescan-badge')!.textContent = String(area.id);
    button.querySelector('strong')!.textContent = REASON[area.reason];
    button.querySelector('small')!.textContent = area.side;
    button.addEventListener('click', () => select(selected === area.id ? null : area));
    list.append(button);
  }
}
