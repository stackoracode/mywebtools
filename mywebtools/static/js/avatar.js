// Avatar PiP extras: live layout sketch, engine status banner, check & preview actions.
import { api, errorText, fmtDuration, h, state, t, toast } from "./core.js";

const CANVASES = {
  shorts: [1080, 1920], shorts_720: [720, 1280], portrait_45: [1080, 1350], square: [1080, 1080],
  landscape: [1920, 1080], landscape_720: [1280, 720],
};
const SAFE = { top: 0.08, bottom: 0.21, left: 0.03, right: 0.14 }; // keep in sync with composite.SHORTS_SAFE
const even = (v) => Math.max(2, Math.round(v / 2) * 2);

// ---------------------------------------------------------------- engine banner
let lastHealth = null;
export async function checkEngine({ quiet = false } = {}) {
  if (!state.tools.some((x) => x.id === "avatar_pip")) return null;
  try { lastHealth = await api("/api/duix/health"); } catch { lastHealth = null; }
  state.health = lastHealth;
  renderBanner(quiet);
  return lastHealth;
}

function renderBanner() {
  const banner = document.getElementById("banner");
  const hlth = lastHealth;
  const route = location.hash || "#/";
  const relevant = route === "#/" || route.startsWith("#/tool/avatar_pip");
  if (!hlth || hlth.ok || !relevant) { banner.hidden = true; return; }
  let message;
  if (hlth.engine === "duix_cloud") message = t("engine.no_key");
  else if (!hlth.reachable) message = t("engine.offline", { url: hlth.url });
  else message = t("engine.share_missing", { path: hlth.share?.path || "", windows_dir: hlth.windows_dir || "" });
  banner.className = `banner${hlth.reachable ? "" : " bad"}`;
  banner.replaceChildren(
    h("p", {}, h("b", {}, t("engine.title")), " ", message),
    h("button", { class: "btn small", onclick: () => checkEngine() }, t("engine.retry")),
    hlth.wol && !hlth.reachable ? h("button", { class: "btn small", onclick: async () => {
      try { await api("/api/duix/wake", { method: "POST" }); toast(t("engine.wake_sent")); setTimeout(checkEngine, 30000); }
      catch (e) { toast(errorText(e.error)); }
    } }, t("engine.wake")) : null,
    h("a", { class: "btn small", href: "#/admin/engine" }, t("engine.settings")));
  banner.hidden = false;
}

export function startEngineWatch() {
  checkEngine();
  setInterval(() => { if (document.visibilityState === "visible") checkEngine({ quiet: true }); }, 60000);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") checkEngine(); });
  window.addEventListener("hashchange", renderBanner);
}

// ---------------------------------------------------------------- geometry (mirror of composite.py)
export function geometry(v, main, src) {
  let [cw, ch] = CANVASES[v.canvas] || (v.canvas === "custom" ? [even(v.canvas_w), even(v.canvas_h)] : [main?.width || 1080, main?.height || 1920]);
  const mw = main?.width || cw, mh = main?.height || ch;
  let fit = v.fit;
  if (fit === "auto") { const r = (mw / mh) / (cw / ch); fit = r >= 0.9 && r <= 1.12 ? "cover" : "blur"; }
  let mainRect;
  if (fit === "cover") { const s = Math.max(cw / mw, ch / mh); mainRect = [(cw - mw * s) / 2, (ch - mh * s) / 2, mw * s, mh * s]; }
  else {
    const s = Math.min(cw / mw, ch / mh), sw = mw * s, sh = mh * s;
    const y = v.main_align === "top" ? 0 : v.main_align === "bottom" ? ch - sh : (ch - sh) / 2;
    mainRect = [(cw - sw) / 2, y, sw, sh];
  }
  const w = even(Math.min(Math.max(v.size, 32), cw));
  let hgt = w;
  if (v.size_h > 0 && v.shape !== "circle") hgt = even(v.size_h);
  else if (v.crop_mode === "full" && src?.width) hgt = even(w * src.height / src.width);
  hgt = Math.min(hgt, ch);
  let [l, r, tp, b] = [0, 0, 0, 0];
  if (v.safe_area && ch > cw) { l = SAFE.left * cw; r = SAFE.right * cw; tp = SAFE.top * ch; b = SAFE.bottom * ch; }
  const [vert, horiz] = v.anchor === "center" ? ["middle", "center"] : v.anchor.split("_");
  let x = horiz === "left" ? l + v.margin_x : horiz === "right" ? cw - r - v.margin_x - w : (cw - w) / 2;
  let y = vert === "top" ? tp + v.margin_y : vert === "bottom" ? ch - b - v.margin_y - hgt : (ch - hgt) / 2;
  x = Math.min(Math.max(x, 0), cw - w); y = Math.min(Math.max(y, 0), ch - hgt);
  return { cw, ch, fit, mainRect, box: [w, hgt], pos: [Math.round(x), Math.round(y)], safe: ch > cw };
}

function cropRect(v, src, bw, bh) {
  const sw = src?.width, sh = src?.height;
  if (!sw || v.crop_mode === "full") return [0, 0, 1, 1];
  let cx = v.crop_cx, cy = v.crop_cy, zoom = v.crop_zoom;
  if (v.crop_mode === "auto") { cx = 0.5; zoom = 1; cy = sh > sw ? 0.39 : 0.45; }
  const a = bw / bh;
  let cw, ch;
  if (sw / sh > a) { ch = sh; cw = ch * a; } else { cw = sw; ch = cw / a; }
  cw /= Math.max(zoom, 1); ch /= Math.max(zoom, 1);
  const x = Math.min(Math.max(cx * sw - cw / 2, 0), sw - cw), y = Math.min(Math.max(cy * sh - ch / 2, 0), sh - ch);
  return [x / sw, y / sh, cw / sw, ch / sh];
}

function shapePath(ctx, shape, x, y, w, hh, radius) {
  const cx = x + w / 2, cy = y + hh / 2, rx = w / 2, ry = hh / 2;
  ctx.beginPath();
  if (shape === "circle") { ctx.arc(cx, cy, Math.min(rx, ry), 0, Math.PI * 2); }
  else if (shape === "ellipse") { ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2); }
  else if (shape === "rect") { ctx.rect(x, y, w, hh); }
  else if (shape === "rounded") { ctx.roundRect ? ctx.roundRect(x, y, w, hh, Math.min(radius, rx, ry)) : ctx.rect(x, y, w, hh); }
  else {
    let pts = [];
    if (shape === "diamond") pts = [[cx, y], [x + w, cy], [cx, y + hh], [x, cy]];
    else if (shape === "hexagon") for (let a = -90; a < 270; a += 60) pts.push([cx + rx * Math.cos(a * Math.PI / 180), cy + ry * Math.sin(a * Math.PI / 180)]);
    else if (shape === "squircle") for (let i = 0; i < 120; i++) { const a = 2 * Math.PI * i / 120, c = Math.cos(a), s = Math.sin(a); pts.push([cx + rx * Math.sign(c) * Math.sqrt(Math.abs(c)), cy + ry * Math.sign(s) * Math.sqrt(Math.abs(s))]); }
    else if (shape === "heart") {
      const raw = [];
      for (let i = 0; i < 120; i++) { const a = 2 * Math.PI * i / 120; raw.push([16 * Math.sin(a) ** 3, -(13 * Math.cos(a) - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a))]); }
      const xs = raw.map((p) => p[0]), ys = raw.map((p) => p[1]);
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      pts = raw.map(([px, py]) => [x + (px - x0) / (x1 - x0) * w, y + (py - y0) / (y1 - y0) * hh]);
    }
    pts.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
  }
  ctx.closePath();
}

// ---------------------------------------------------------------- form enhancement
export function enhanceAvatarForm(ctrl) {
  const v = ctrl.values;
  const media = { main: null, model: null, clip: null, modelThumb: null, clipThumb: null };
  const canvas = h("canvas", { width: 220, height: 391, "aria-label": t("avatar.layout") });
  const info = h("div", { class: "muted", style: "font-size:.85rem" });
  const output = h("div", { class: "stack" });
  const checkBtn = h("button", { type: "button", class: "btn small", onclick: runCheck }, t("avatar.check"));
  const previewBtn = h("button", { type: "button", class: "btn small", onclick: runPreview }, t("avatar.preview_frame"));
  const card = h("section", { class: "card stack" },
    h("h3", { style: "margin:0" }, t("avatar.layout")),
    h("div", { class: "pip-preview" }, canvas,
      h("div", { class: "side" }, info, h("div", { class: "row" }, checkBtn, previewBtn), output)));
  ctrl.form.querySelector(".submit-bar").before(card);

  const draw = () => {
    const src = v.avatar_source === "clip" ? media.clip : media.model;
    const g = geometry(v, media.main, src);
    const W = 220, scale = W / g.cw, H = Math.round(g.ch * scale);
    canvas.width = W * devicePixelRatio; canvas.height = H * devicePixelRatio;
    canvas.style.aspectRatio = `${g.cw} / ${g.ch}`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(devicePixelRatio * scale, 0, 0, devicePixelRatio * scale, 0, 0);
    ctx.fillStyle = g.fit === "color" ? v.bg_color : g.fit === "blur" ? "#4a5263" : "#2a2f3a";
    ctx.fillRect(0, 0, g.cw, g.ch);
    const [mx, my, mw, mh] = g.mainRect;
    const grad = ctx.createLinearGradient(0, my, 0, my + mh);
    grad.addColorStop(0, "#7d8fb3"); grad.addColorStop(1, "#3c4a66");
    ctx.fillStyle = grad; ctx.fillRect(mx, my, mw, mh);
    ctx.fillStyle = "rgba(255,255,255,.7)"; ctx.font = `${Math.round(g.cw / 18)}px system-ui`; ctx.textAlign = "center";
    ctx.fillText(t("avatar.main_video"), mx + mw / 2, my + mh / 2);
    if (v.safe_area && g.safe) {
      ctx.fillStyle = "rgba(255,80,80,.18)";
      ctx.fillRect(0, 0, g.cw, SAFE.top * g.ch);
      ctx.fillRect(0, g.ch * (1 - SAFE.bottom), g.cw, SAFE.bottom * g.ch);
      ctx.fillRect(g.cw * (1 - SAFE.right), 0, SAFE.right * g.cw, g.ch);
    }
    const [x, y] = g.pos, [w, hh] = g.box;
    ctx.save();
    if (v.shadow) { ctx.shadowColor = `rgba(0,0,0,${v.shadow_opacity / 100})`; ctx.shadowBlur = v.shadow_blur; ctx.shadowOffsetY = v.shadow_blur / 3; }
    shapePath(ctx, v.shape, x, y, w, hh, v.radius);
    ctx.fillStyle = "#f0b46b"; ctx.globalAlpha = v.opacity / 100; ctx.fill();
    ctx.restore();
    const img = v.avatar_source === "clip" ? media.clipThumb : media.modelThumb;
    if (img?.complete && img.naturalWidth) {
      const [fx, fy, fw, fh] = cropRect(v, src, w, hh);
      ctx.save(); shapePath(ctx, v.shape, x, y, w, hh, v.radius); ctx.clip(); ctx.globalAlpha = v.opacity / 100;
      if (v.mirror) { ctx.translate(x * 2 + w, 0); ctx.scale(-1, 1); }
      ctx.drawImage(img, fx * img.naturalWidth, fy * img.naturalHeight, fw * img.naturalWidth, fh * img.naturalHeight, x, y, w, hh);
      ctx.restore();
    }
    if (v.border_w > 0) {
      ctx.save(); shapePath(ctx, v.shape, x + v.border_w / 2, y + v.border_w / 2, w - v.border_w, hh - v.border_w, Math.max(0, v.radius - v.border_w / 2));
      ctx.lineWidth = v.border_w; ctx.strokeStyle = v.border_color; ctx.stroke(); ctx.restore();
    }
    const lines = [`${g.cw}×${g.ch} · ${t(`tool.avatar_pip.o.fit.${g.fit}`, {}, g.fit)}`, `${t("avatar.avatar")}: ${w}×${hh} @ ${x},${y}`];
    if (media.main) lines.push(`${t("avatar.main_video")}: ${media.main.width}×${media.main.height} · ${fmtDuration(media.main.duration)}${media.main.is_hdr ? " · HDR" : ""}`);
    if (src) lines.push(`${t("avatar.avatar")}: ${src.width}×${src.height}${src.duration ? ` · ${fmtDuration(src.duration)}` : ""}`);
    info.replaceChildren(...lines.map((l) => h("div", {}, l)));
  };

  const loadThumb = (item, lib) => {
    if (!item?.has_thumb) return null;
    const img = new Image();
    img.onload = draw;
    img.src = `/api/library/${lib}/${item.id}/thumb`;
    return img;
  };

  const prevChange = ctrl.hooks?.onChange;
  ctrl.hooks.onChange = (...a) => { prevChange?.(...a); draw(); };
  ctrl.hooks.onFilesChanged = async (name, list) => {
    if (name !== "main_video") return;
    const done = list.find((f) => f.id);
    media.main = done ? await api(`/api/uploads/${done.id}/probe`, { method: "POST" }).catch(() => null) : null;
    draw();
  };
  ctrl.hooks.onLibraryChanged = (name, item) => {
    if (name === "model") { media.model = item?.info || null; media.modelThumb = loadThumb(item, "avatar_models"); }
    if (name === "clip") { media.clip = item?.info || null; media.clipThumb = loadThumb(item, "avatar_clips"); }
    draw();
  };
  ctrl.hooks.beforeSubmit = async () => {
    if (v.avatar_source !== "duix") return true;
    const hlth = await checkEngine();
    if (hlth && !hlth.ok) return confirm(t("avatar.offline_confirm"));
    return true;
  };

  async function runCheck() {
    checkBtn.disabled = true;
    try {
      const res = await api("/api/tools/avatar_pip/actions/check", { method: "POST", body: { params: v, files: ctrl.fileIds() } });
      const rows = [];
      if (res.voice_seconds) rows.push(t("avatar.voice_len", { s: res.voice_seconds }));
      const acc = res.account;
      if (acc?.plan === "local") rows.push(`✓ ${t("engine.local_ok", { url: acc.url, ms: acc.latency_ms ?? "?" })}`);
      else if (acc) rows.push(t("avatar.credits", { need: res.credits_needed, left: acc.credits_left, plan: acc.plan }));
      if (res.account_error) rows.push(`✕ ${errorText(res.account_error)}`);
      output.replaceChildren(...rows.map((r) => h("div", {}, r)),
        res.warnings?.length ? h("ul", { class: "warn-list" }, ...res.warnings.map((w) => h("li", {}, t(`warn.${w}`, {}, t(`err.${w}`, {}, w))))) : "");
      ctrl.slot("credits_info")?.replaceChildren(...rows.map((r) => h("div", { class: "help" }, r)));
    } catch (e) { toast(errorText(e.error)); } finally { checkBtn.disabled = false; }
  }

  async function runPreview() {
    previewBtn.disabled = true;
    previewBtn.textContent = "…";
    try {
      const res = await api("/api/tools/avatar_pip/actions/preview", { method: "POST", body: { params: v, files: ctrl.fileIds() } });
      output.replaceChildren(h("img", { class: "preview", src: `${res.image}?t=${Date.now()}`, alt: t("avatar.preview_frame") }),
        ...(res.notes || []).map((n) => h("small", { class: "muted" }, t(`note.${n}`, {}, n))));
    } catch (e) { toast(errorText(e.error)); } finally { previewBtn.disabled = false; previewBtn.textContent = t("avatar.preview_frame"); }
  }

  draw();
  return draw;
}
