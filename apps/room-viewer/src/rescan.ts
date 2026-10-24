import type { RescanArea, RescanOverlay } from '../../../packages/room-runtime/splat-room';
import { RESCAN_COLORS } from '../../../packages/room-runtime/splat-room-colors';
import type { V3 } from '../../../packages/room-runtime/types';

const REASON = { inaccurate: 'Blurry or smeared', 'few-photos': 'Too few photos' } as const;
/** areas listed before "Show all" */
const SHORT_LIST = 4;

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

function details(area: RescanArea, rescan: RescanOverlay) {
  const detail = document.createElement('div');
  detail.className = 'rescan-detail';
  const text = document.createElement('p');
  text.textContent = area.text;
  detail.append(text);
  const view = area.view === null ? undefined : rescan.review.views[area.view];
  if (view) {
    const figures = document.createElement('div');
    figures.className = 'rescan-compare';
    for (const [src, caption] of [[view.photo, 'Your photo'], [view.render, 'Photoreal result']]) {
      const figure = document.createElement('figure');
      const img = document.createElement('img');
      img.src = src; img.alt = `${caption}, spot ${area.id}`;
      const cap = document.createElement('figcaption');
      cap.textContent = caption;
      figure.append(img, cap);
      figures.append(figure);
    }
    detail.append(figures);
  } else {
    const note = document.createElement('p');
    note.className = 'help';
    note.textContent = 'No held-out photo sees this spot, so there is nothing to compare. The green ring marks where to stand.';
    detail.append(note);
  }
  return detail;
}

/** The Rescan sidebar section: weak spots of a photoreal scene, where to stand, and the real
 * photo beside the result. Markers stay hidden until asked for, so the scene reads first.
 * Hidden for rooms without a review. */
export function showRescan(host: HTMLElement, rescan: RescanOverlay | undefined,
  actions: { focus(position: V3, target: V3): void; redraw(): void }) {
  host.hidden = !rescan;
  const list = host.querySelector<HTMLElement>('.rescan-list')!;
  const more = host.querySelector<HTMLButtonElement>('.rescan-more')!;
  const summary = host.querySelector<HTMLElement>('#rescan-summary')!;
  const intro = host.querySelector<HTMLElement>('.rescan-intro')!;
  const show = host.querySelector<HTMLInputElement>('#rescan-show')!;
  const toggle = show.closest<HTMLElement>('label')!;
  const legend = host.querySelector<HTMLElement>('.rescan-legend')!;
  list.replaceChildren();
  more.hidden = true;
  if (!rescan) return;
  const { review } = rescan;
  const count = review.areas.length;
  summary.textContent = count ? `${count} spot${count === 1 ? '' : 's'}` : 'All good';
  summary.title = `${Math.round(review.weakFraction * 100)}% of scanned surfaces are weak`;
  intro.textContent = count ? 'Spots to photograph again. Pick one to see where to stand.'
    : 'Every surface has enough sharp photos. Nothing to rescan.';
  toggle.hidden = legend.hidden = !count;
  rescan.group.visible = show.checked = false;
  rescan.highlight(null);
  show.onchange = () => {
    rescan.group.visible = show.checked;
    if (!show.checked) select(null);
    actions.redraw();
  };

  let selected: number | null = null;
  let expanded = false;
  const buttons = review.areas.map(area => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.area = String(area.id);
    button.setAttribute('aria-expanded', 'false');
    button.style.setProperty('--marker', RESCAN_COLORS[area.reason]);
    button.innerHTML = '<span class="rescan-badge"></span><span class="rescan-label"><strong></strong><small></small></span>';
    button.querySelector('.rescan-badge')!.textContent = String(area.id);
    button.querySelector('strong')!.textContent = REASON[area.reason];
    button.querySelector('small')!.textContent = area.side;
    button.addEventListener('click', () => select(selected === area.id ? null : area));
    return button;
  });

  function render() {
    const visible = expanded ? buttons : buttons.slice(0, SHORT_LIST);
    const nodes: HTMLElement[] = [];
    for (const button of visible) {
      nodes.push(button);
      const area = review.areas.find(a => a.id === Number(button.dataset.area))!;
      button.setAttribute('aria-expanded', String(area.id === selected));
      if (area.id === selected) nodes.push(details(area, rescan!));
    }
    list.replaceChildren(...nodes);
    more.hidden = buttons.length <= SHORT_LIST;
    more.textContent = expanded ? 'Show fewer' : `Show all ${buttons.length}`;
  }

  function select(area: RescanArea | null) {
    selected = area?.id ?? null;
    rescan!.highlight(selected);
    if (area) {
      if (!expanded && review.areas.indexOf(area) >= SHORT_LIST) expanded = true;
      rescan!.group.visible = show.checked = true;
      actions.focus(...frame(area));
    }
    render();
    actions.redraw();
  }

  more.onclick = () => { expanded = !expanded; render(); };
  render();
}
