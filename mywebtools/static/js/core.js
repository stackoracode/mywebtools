// Shared helpers: API client, i18n, uploads, small DOM utilities.

export const state = { meta: null, tools: [], jobs: new Map(), lang: "en", dict: {}, fallback: {}, health: null };

// ---------- i18n ----------
const LANG_KEY = "mwt.lang";

export function pickLanguage(available) {
  let saved = null;
  try { saved = localStorage.getItem(LANG_KEY); } catch { /* private mode */ }
  if (saved && available[saved]) return saved;
  const nav = (navigator.language || "en").toLowerCase();
  if (nav.startsWith("zh") && available["zh-CN"]) return "zh-CN";
  return available[state.meta?.default_language] ? state.meta.default_language : "en";
}

export async function loadLanguage(lang) {
  const get = async (l) => (await fetch(`/static/i18n/${l}.json`, { cache: "no-cache" })).json().catch(() => ({}));
  state.fallback = await get("en");
  state.dict = lang === "en" ? state.fallback : await get(lang);
  state.lang = lang;
  document.documentElement.lang = lang;
  try { localStorage.setItem(LANG_KEY, lang); } catch { /* ignore */ }
  translateStatic();
}

/** t("err.duix.too_long", {seconds: 3}, "English fallback") */
export function t(key, params = {}, fallback) {
  let text = state.dict[key] ?? state.fallback[key] ?? fallback ?? key;
  for (const [k, v] of Object.entries(params || {})) text = text.replaceAll(`{${k}}`, String(v));
  return text;
}

export function translateStatic(root = document) {
  root.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n, {}, el.textContent); });
  root.querySelectorAll("[data-i18n-placeholder]").forEach((el) => { el.placeholder = t(el.dataset.i18nPlaceholder, {}, el.placeholder); });
}

const displayNames = {};
export function languageName(code) {
  if (!code || code === "auto") return code;
  const bcp = state.meta?.tesseract_bcp47?.[code] || code;
  try {
    displayNames[state.lang] ??= new Intl.DisplayNames([state.lang], { type: "language" });
    const name = displayNames[state.lang].of(bcp);
    return name && name !== bcp ? `${name} (${code})` : code;
  } catch { return code; }
}

export function errorText(error) {
  if (!error) return "";
  const msg = t(`err.${error.code}`, error.params, error.code);
  return error.detail ? `${msg} — ${error.detail}` : msg;
}

// ---------- API ----------
export class ApiError extends Error {
  constructor(error, status) { super(error?.code || "http"); this.error = error; this.status = status; }
}

let onUnauthorized = () => {};
export function setUnauthorizedHandler(fn) { onUnauthorized = fn; }

export async function api(path, { method = "GET", body, signal } = {}) {
  const init = { method, credentials: "same-origin", signal, headers: {} };
  if (body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
  let resp;
  try { resp = await fetch(path, init); } catch (e) { throw new ApiError({ code: "network", params: {}, detail: e.message }, 0); }
  const data = await resp.json().catch(() => ({}));
  if (resp.status === 401) { onUnauthorized(); throw new ApiError(data.error || { code: "auth.required" }, 401); }
  if (!resp.ok) throw new ApiError(data.error || { code: "http", params: {}, detail: String(resp.status) }, resp.status);
  return data;
}

/** Raw-body upload with progress (no multipart: fastest path on a small board). */
export function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `/api/uploads?name=${encodeURIComponent(file.name)}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      if (xhr.status === 401) onUnauthorized();
      xhr.status < 300 ? resolve(data) : reject(new ApiError(data.error || { code: "upload.failed" }, xhr.status));
    };
    xhr.onerror = () => reject(new ApiError({ code: "network", params: {}, detail: "upload" }, 0));
    xhr.send(file);
    file._xhr = xhr;
  });
}

// ---------- live events ----------
let source;
const listeners = new Set();
export function onJobEvent(fn) { listeners.add(fn); return () => listeners.delete(fn); }

export function connectEvents() {
  source?.close();
  source = new EventSource("/api/events");
  source.onmessage = (msg) => {
    let event;
    try { event = JSON.parse(msg.data); } catch { return; }
    if (event.type === "job") state.jobs.set(event.job.id, event.job);
    if (event.type === "deleted") state.jobs.delete(event.id);
    listeners.forEach((fn) => fn(event));
  };
}

export async function refreshJobs() {
  const data = await api("/api/jobs?limit=200");
  state.jobs = new Map(data.jobs.map((j) => [j.id, j]));
  listeners.forEach((fn) => fn({ type: "sync" }));
  return data;
}

// ---------- DOM ----------
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

let toastTimer;
export function toast(message, ms = 3200) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

export function fmtBytes(n) {
  if (n === null || n === undefined) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i ? 1 : 0)} ${units[i]}`;
}

export function fmtDuration(sec) {
  if (!sec && sec !== 0) return "";
  sec = Math.round(sec);
  const m = Math.floor(sec / 60), s = sec % 60;
  return m ? `${m}:${String(s).padStart(2, "0")}` : `${s}s`;
}

export function fmtTime(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleString(state.lang, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function toolName(id) {
  const tool = state.tools.find((x) => x.id === id);
  return t(`tool.${id}.name`, {}, tool?.name || id);
}

export const storage = {
  get(key, fallback = null) { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* quota / private mode */ } },
  remove(key) { try { localStorage.removeItem(key); } catch { /* ignore */ } },
};
