// Management UI: avatar engine, model services, AI model, runtime settings, libraries, system status, security.
import { api, errorText, fmtBytes, fmtDuration, h, state, t, toast } from "./core.js";
import { checkEngine } from "./avatar.js";

const TABS = ["engine", "nodes", "ai", "settings", "libraries", "system", "security"];
const AI_FIELDS = ["llm_provider", "llm_base_url", "llm_model", "llm_reasoning_effort", "llm_timeout_seconds",
  "llm_transcribe_model", "llm_tts_model", "llm_tts_voice", "subtitle_font"];
// model services on the GPU PC; folders are paths on that PC
const NODE_FIELDS = {
  whisper: ["node_whisper_url", "whisper_model_dir", "whisper_device", "whisper_compute_type", "whisper_beam_size"],
  tts: ["node_tts_url", "indextts_model_dir", "indextts_device", "indextts_bf16"],
  align: ["node_align_url", "mms_model_dir", "mms_device"],
};
const SPEECH_FIELDS = ["asr_engine", "tts_engine", "align_engine", "node_timeout_seconds", "node_free_vram", "whisper_model"];
const MODEL_FIELDS = [...Object.values(NODE_FIELDS).flat(), ...SPEECH_FIELDS];
const ENGINE_LOCAL = ["duix_local_url", "duix_windows_dir", "duix_share_path", "duix_timeout_seconds", "duix_poll_seconds",
  "duix_default_model", "duix_cleanup_share", "duix_wol_mac"];
const ENGINE_CLOUD = ["duix_base_url", "duix_upload_height", "duix_upload_fps", "duix_delete_remote"];
const SELECTS = {
  video_encoder: ["libx264", "h264_rkmpp", "h264_v4l2m2m"],
  x264_preset: ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"],
  default_language: () => Object.keys(state.meta.ui_languages),
  avatar_engine: ["duix_local", "duix_cloud"],
  llm_provider: ["openai", "openai_compatible"],
  llm_reasoning_effort: () => state.meta.llm?.efforts || ["", "low", "medium", "high", "xhigh"],
  asr_engine: ["auto", "node", "local", "api"],
  tts_engine: ["indextts", "openai"],
  align_engine: ["auto", "mms", "basic"],
  whisper_model: ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"],
  whisper_device: ["auto", "cuda", "cpu"],
  indextts_device: ["auto", "cuda", "cpu"],
  mms_device: ["auto", "cuda", "cpu"],
  whisper_compute_type: ["auto", "float16", "int8_float16", "int8"],
};

export async function renderAdmin(view, tab = "engine") {
  if (!TABS.includes(tab)) tab = "engine";
  const body = h("div", { class: "stack" });
  view.replaceChildren(
    h("h1", {}, t("nav.admin")),
    h("div", { class: "tabs", role: "tablist" }, ...TABS.map((name) => h("a", {
      class: "btn small", role: "tab", href: `#/admin/${name}`, "aria-selected": String(name === tab) }, t(`admin.${name}`)))),
    body);
  try {
    await ({ engine, nodes, ai, settings, libraries, system, security })[tab](body);
  } catch (e) { body.replaceChildren(h("p", { class: "error" }, errorText(e.error) || String(e))); }
}

function settingField(f, values, onChange) {
  const label = t(`set.${f.name}`, {}, f.name);
  const help = t(`set.${f.name}.help`, {}, "");
  const id = `s-${f.name}`;
  let input;
  const opts = typeof SELECTS[f.name] === "function" ? SELECTS[f.name]() : SELECTS[f.name];
  if (f.type === "bool") {
    input = h("input", { type: "checkbox", id, checked: !!values[f.name], onchange: (e) => onChange(f.name, e.target.checked) });
    return h("div", { class: "field" }, h("label", { class: "switch", for: id }, h("span", {}, label), input), help && h("div", { class: "help" }, help));
  }
  if (opts) {
    input = h("select", { id, onchange: (e) => onChange(f.name, e.target.value) },
      ...opts.map((o) => h("option", { value: o, selected: o === values[f.name] },
        t(`set.${f.name}.${o || "default"}`, {}, state.meta.ui_languages?.[o] || o || "—"))));
  } else {
    input = h("input", { id, type: f.type === "int" ? "number" : "text", inputmode: f.type === "int" ? "numeric" : undefined,
      value: values[f.name] ?? "", placeholder: f.placeholder || "", list: f.list || undefined,
      onchange: (e) => onChange(f.name, f.type === "int" ? Number(e.target.value) : e.target.value) });
  }
  return h("div", { class: "field" }, h("label", { for: id }, label), input, help && h("div", { class: "help" }, help));
}

// ------------------------------------------------------------------ engine
async function engine(body) {
  const data = await api("/api/settings");
  const values = { ...data.values };
  const fields = Object.fromEntries(data.fields.map((f) => [f.name, f]));
  const models = (await api("/api/library/avatar_models").catch(() => ({ items: [] }))).items;
  let key = "";
  const status = h("div", { class: "stack" });
  const holder = h("div", { class: "stack" });

  const draw = () => {
    const local = values.avatar_engine !== "duix_cloud";
    const grid = h("div", { class: "group-body", style: "padding:0" });
    grid.append(settingField(fields.avatar_engine, values, (k, v) => { values[k] = v; draw(); }));
    for (const name of local ? ENGINE_LOCAL : ENGINE_CLOUD) {
      if (name === "duix_default_model") {
        grid.append(h("div", { class: "field" }, h("label", { for: "s-model" }, t("set.duix_default_model")),
          h("select", { id: "s-model", onchange: (e) => { values.duix_default_model = e.target.value; } },
            h("option", { value: "" }, models.length ? t("ui.choose") : t("ui.library_empty")),
            ...models.map((m) => h("option", { value: m.id, selected: m.id === values.duix_default_model }, m.name))),
          h("div", { class: "help" }, t("set.duix_default_model.help"))));
        continue;
      }
      const f = { ...fields[name] };
      if (name === "duix_share_path") f.placeholder = state.meta.os === "nt" ? values.duix_windows_dir : "/mnt/duix_temp";
      grid.append(settingField(f, values, (k, v) => { values[k] = v; }));
    }
    if (!local) {
      grid.append(h("div", { class: "field wide" }, h("label", { for: "s-key" }, t("set.DUIX_API_KEY")),
        h("input", { id: "s-key", type: "password", autocomplete: "off", placeholder: data.secrets.DUIX_API_KEY.configured
          ? `${t("set.configured")} ${data.secrets.DUIX_API_KEY.hint}` : "", oninput: (e) => { key = e.target.value.trim(); } }),
        h("div", { class: "help" }, t("set.DUIX_API_KEY.help"))));
    }
    holder.replaceChildren(
      h("section", { class: "card stack" }, h("h2", { style: "margin:0" }, t("admin.engine_title")),
        h("p", { class: "muted", style: "margin:0" }, t(local ? "admin.engine_local_intro" : "admin.engine_cloud_intro")), grid,
        h("div", { class: "row" },
          h("button", { class: "btn", onclick: test }, t("admin.test_connection")),
          h("button", { class: "btn primary", onclick: save }, t("ui.save")))),
      h("section", { class: "card" }, status));
  };

  async function test() {
    status.replaceChildren(h("p", {}, "…"));
    const local = values.avatar_engine !== "duix_cloud";
    const res = await api("/api/duix/verify", { method: "POST", body: local
      ? { engine: "duix_local", url: values.duix_local_url, share_path: values.duix_share_path || undefined }
      : { engine: "duix_cloud", key } }).catch((e) => ({ ok: false, error: e.error }));
    status.replaceChildren(...(local ? localStatus(res) : cloudStatus(res)));
  }

  async function save() {
    try {
      await api("/api/settings", { method: "PUT", body: { values } });
      if (key) await api("/api/secrets", { method: "PUT", body: { DUIX_API_KEY: key } });
      toast(t("ui.saved"));
      checkEngine();
      test();
    } catch (e) { toast(errorText(e.error)); }
  }

  draw();
  body.replaceChildren(holder);
  test();
}

function dot(ok) { return h("span", { class: `status-dot ${ok ? "ok" : "bad"}` }); }

function localStatus(res) {
  const share = res.share || {};
  return [
    h("h3", { style: "margin-top:0" }, t("admin.status")),
    h("div", {}, dot(res.reachable), res.reachable ? t("engine.local_ok", { url: res.url, ms: res.latency_ms }) : t("engine.offline", { url: res.url || "" })),
    h("div", {}, dot(share.writable), share.writable ? t("engine.share_ok", { path: share.path }) :
      t("engine.share_missing", { path: share.path || "", windows_dir: res.windows_dir || "" })),
    res.error ? h("small", { class: "muted" }, errorText(res.error)) : null,
    !res.ok ? h("details", {}, h("summary", {}, t("admin.howto")), h("ol", {}, ...[1, 2, 3, 4].map((n) => h("li", {}, t(`admin.howto.${n}`, { windows_dir: res.windows_dir || "" }))))) : null,
  ];
}

function cloudStatus(res) {
  if (!res.valid) return [h("div", {}, dot(false), errorText(res.error) || t("err.duix.auth_failed"))];
  return [
    h("div", {}, dot(true), t("admin.key_valid")),
    h("dl", { class: "kv" },
      h("dt", {}, t("admin.plan")), h("dd", {}, `${res.plan} — ${res.summary}`),
      h("dt", {}, t("admin.credits")), h("dd", {}, `${res.credits_left} / ${res.credits_total ?? "?"} (≈ ${fmtDuration(res.seconds_left)})`),
      h("dt", {}, t("admin.max_len")), h("dd", {}, `${res.max_minutes} min · ${res.max_resolution}`),
      h("dt", {}, t("admin.cycle")), h("dd", {}, res.cycle)),
  ];
}

// ------------------------------------------------------------------ model services (GPU PC)
async function nodes(body) {
  const data = await api("/api/settings");
  const values = { ...data.values };
  const fields = Object.fromEntries(data.fields.map((f) => [f.name, f]));
  let token = "";
  const statuses = {};
  const set = (k, v) => { values[k] = v; };

  async function check(kind) {
    const box = statuses[kind];
    box.replaceChildren(h("p", { class: "muted" }, "…"));
    const [urlKey, dirKey] = NODE_FIELDS[kind];
    const res = await api(`/api/nodes/${kind}/check`, { method: "POST", body: { url: values[urlKey], model_dir: values[dirKey] } })
      .catch((e) => ({ reachable: false, error: e.error }));
    box.replaceChildren(...nodeStatus(kind, res));
  }

  async function save(quiet = false) {
    try {
      await api("/api/settings", { method: "PUT", body: { values } });
      if (token) await api("/api/secrets", { method: "PUT", body: { NODE_TOKEN: token } });
      if (!quiet) toast(t("ui.saved"));
      return true;
    } catch (e) { toast(errorText(e.error)); return false; }
  }

  async function act(kind, action) {
    if (!(await save(true))) return;
    statuses[kind].prepend(h("p", { class: "muted" }, t(`node.${action}ing`)));
    try {
      const res = await api(`/api/nodes/${kind}/${action}`, { method: "POST" });
      toast(action === "load" ? t("node.loaded_in", { s: res.seconds }) : t("node.unloaded"));
    } catch (e) { toast(errorText(e.error)); }
    check(kind);
  }

  const cards = Object.keys(NODE_FIELDS).map((kind) => {
    statuses[kind] = h("div", { class: "stack" });
    return h("section", { class: "card stack" },
      h("h2", { style: "margin:0" }, t(`node.${kind}`)),
      h("p", { class: "muted", style: "margin:0" }, t(`node.${kind}.intro`)),
      h("div", { class: "group-body", style: "padding:0" }, ...NODE_FIELDS[kind].map((name) => settingField(fields[name], values, set))),
      h("div", { class: "row" },
        h("button", { class: "btn", onclick: () => check(kind) }, t("node.check")),
        h("button", { class: "btn", onclick: () => act(kind, "load") }, t("node.load")),
        h("button", { class: "btn", onclick: () => act(kind, "unload") }, t("node.unload"))),
      statuses[kind]);
  });
  const engines = h("section", { class: "card stack" },
    h("h2", { style: "margin:0" }, t("node.engines")),
    h("p", { class: "muted", style: "margin:0" }, t("node.engines_intro")),
    h("div", { class: "group-body", style: "padding:0" },
      ...SPEECH_FIELDS.map((name) => settingField(fields[name], values, set)),
      h("div", { class: "field" }, h("label", { for: "s-node-token" }, t("set.NODE_TOKEN")),
        h("input", { id: "s-node-token", type: "password", autocomplete: "off",
          placeholder: data.secrets.NODE_TOKEN?.configured ? `${t("set.configured")} ${data.secrets.NODE_TOKEN.hint}` : "",
          oninput: (e) => { token = e.target.value.trim(); } }),
        h("div", { class: "help" }, t("set.NODE_TOKEN.help")))),
    h("div", { class: "row" }, h("span", { class: "spacer" }),
      h("button", { class: "btn primary", onclick: async () => { if (await save()) Object.keys(NODE_FIELDS).forEach(check); } }, t("ui.save"))));
  const howto = h("details", { class: "card" }, h("summary", {}, t("node.howto")),
    h("ol", {}, ...[1, 2, 3, 4].map((n) => h("li", {}, t(`node.howto.${n}`)))),
    h("pre", { class: "log" }, "powershell -ExecutionPolicy Bypass -File services\\install.ps1\npowershell -ExecutionPolicy Bypass -File services\\start.ps1"));
  body.replaceChildren(h("p", { class: "muted" }, t("node.intro")), engines, ...cards, howto);
  Object.keys(NODE_FIELDS).forEach(check);
}

function nodeStatus(kind, res) {
  if (!res.reachable) {
    return [h("div", {}, dot(false), t("node.offline", { url: res.url || "" })), res.error ? h("small", { class: "muted" }, errorText(res.error)) : null];
  }
  const model = res.model || {};
  const rows = [
    h("div", {}, dot(true), t("node.online", { version: res.version || "?", host: res.host || "", python: res.python || "" })),
    res.gpu?.length ? h("small", { class: "muted" }, res.gpu.map((g) => t("node.gpu", { name: g.name, free: (g.free_mb / 1024).toFixed(1), total: (g.total_mb / 1024).toFixed(1) })).join(" · "))
      : h("small", { class: "muted" }, t("node.no_gpu")),
    res.import_error ? h("div", {}, dot(false), t("node.import_error"), h("small", { class: "muted" }, ` ${res.import_error}`)) : null,
    h("div", {}, dot(model.ok), model.ok ? t("node.model_ok", { dir: model.dir }) : model.exists ? t("node.model_incomplete", { dir: model.dir })
      : t(model.error === "model_outside_root" ? "err.node.model_outside_root" : "node.model_missing", { dir: model.dir || res.model_dir || "" })),
  ];
  const missing = (model.files || []).filter((f) => !f.ok);
  if (missing.length) rows.push(h("ul", { class: "warn-list" }, ...missing.map((f) => h("li", {}, f.name))));
  if (model.ok) rows.push(h("details", {}, h("summary", { class: "muted" }, t("node.files", { n: (model.files || []).length })),
    h("ul", {}, ...(model.files || []).map((f) => h("li", {}, `✓ ${f.name} · ${fmtBytes(f.size)}`)))));
  for (const [feature, info] of Object.entries(model.optional || {})) {
    rows.push(h("div", {}, dot(info.ok), t(`node.${kind}.${feature}.${info.ok ? "ok" : "missing"}`)));
  }
  const slot = res.slot || {};
  rows.push(h("small", { class: "muted" }, slot.loaded ? t("node.slot_loaded", { s: slot.load_seconds ?? "?" }) : t("node.slot_idle", { s: slot.idle_unload_seconds ?? 0 })));
  if (res.error) rows.push(h("small", { class: "muted" }, errorText(res.error)));
  return rows;
}

// ------------------------------------------------------------------ AI model
async function ai(body) {
  const data = await api("/api/settings");
  const values = { ...data.values };
  const fields = Object.fromEntries(data.fields.map((f) => [f.name, f]));
  let key = "";
  const lists = { chat: h("datalist", { id: "dl-chat" }), tts: h("datalist", { id: "dl-tts" }), transcribe: h("datalist", { id: "dl-asr" }) };
  const status = h("div", { class: "stack" });
  const listInfo = h("small", { class: "muted" });
  const grid = h("div", { class: "group-body", style: "padding:0" });
  const listFor = { llm_model: "dl-chat", llm_tts_model: "dl-tts", llm_transcribe_model: "dl-asr" };
  for (const name of AI_FIELDS) {
    if (name === "llm_base_url") {
      grid.append(settingField(fields[name], values, (k, v) => { values[k] = v; }));
      grid.append(h("div", { class: "field" }, h("label", { for: "s-llm-key" }, t("set.LLM_API_KEY")),
        h("input", { id: "s-llm-key", type: "password", autocomplete: "off",
          placeholder: data.secrets.LLM_API_KEY.configured ? `${t("set.configured")} ${data.secrets.LLM_API_KEY.hint}` : "sk-…",
          oninput: (e) => { key = e.target.value.trim(); } }),
        h("div", { class: "help" }, t("set.LLM_API_KEY.help"))));
      continue;
    }
    grid.append(settingField({ ...fields[name], list: listFor[name] }, values, (k, v) => { values[k] = v; }));
  }

  async function loadModels() {
    listInfo.textContent = "…";
    try {
      const res = await api("/api/llm/models", { method: "POST", body: { base_url: values.llm_base_url, api_key: key || undefined } });
      for (const [kind, el] of Object.entries(lists)) el.replaceChildren(...(res[kind] || []).map((m) => h("option", { value: m })));
      listInfo.textContent = t("admin.models_loaded", { n: res.all.length, chat: res.chat.length });
    } catch (e) { listInfo.textContent = errorText(e.error); }
  }

  async function test() {
    status.replaceChildren(h("p", {}, "…"));
    const res = await api("/api/llm/test", { method: "POST", body: { base_url: values.llm_base_url, api_key: key || undefined,
      provider: values.llm_provider, model: values.llm_model, effort: values.llm_reasoning_effort } })
      .catch((e) => ({ ok: false, error: e.error }));
    status.replaceChildren(res.ok
      ? h("div", {}, dot(true), t("admin.llm_ok", { model: res.model, effort: res.effort || "—", s: res.seconds }), h("small", { class: "muted" }, ` “${res.reply}”`))
      : h("div", {}, dot(false), errorText(res.error)));
  }

  body.replaceChildren(
    h("section", { class: "card stack" }, h("h2", { style: "margin:0" }, t("admin.ai_title")),
      h("p", { class: "muted", style: "margin:0" }, t("admin.ai_intro")), grid, ...Object.values(lists),
      h("div", { class: "row" },
        h("button", { class: "btn", onclick: loadModels }, t("admin.load_models")), listInfo, h("span", { class: "spacer" }),
        h("button", { class: "btn", onclick: test }, t("admin.test_connection")),
        h("button", { class: "btn primary", onclick: async () => {
          try {
            await api("/api/settings", { method: "PUT", body: { values } });
            if (key) await api("/api/secrets", { method: "PUT", body: { LLM_API_KEY: key } });
            toast(t("ui.saved"));
          } catch (e) { toast(errorText(e.error)); }
        } }, t("ui.save")))),
    h("section", { class: "card" }, status));
  if (data.secrets.LLM_API_KEY.configured) loadModels();
}

// ------------------------------------------------------------------ runtime settings
async function settings(body) {
  const data = await api("/api/settings");
  const values = { ...data.values };
  const grid = h("div", { class: "group-body", style: "padding:0" },
    ...data.fields.filter((f) => !f.name.startsWith("duix_") && f.name !== "avatar_engine" && !AI_FIELDS.includes(f.name)
      && !MODEL_FIELDS.includes(f.name))
      .map((f) => settingField(f, values, (k, v) => { values[k] = v; })));
  body.replaceChildren(h("section", { class: "card stack" }, grid,
    h("button", { class: "btn primary", onclick: async () => {
      try { await api("/api/settings", { method: "PUT", body: { values } }); toast(t("ui.saved")); } catch (e) { toast(errorText(e.error)); }
    } }, t("ui.save"))));
}

// ------------------------------------------------------------------ libraries
async function libraries(body) {
  body.replaceChildren();
  for (const lib of Object.keys(state.meta.libraries)) {
    const list = h("div", { class: "lib-items" });
    const section = h("section", { class: "card stack" }, h("h2", { style: "margin:0" }, t(`lib.${lib}`)),
      h("p", { class: "muted", style: "margin:0" }, t(`lib.${lib}.help`)), list);
    body.append(section);
    const load = async () => {
      const items = (await api(`/api/library/${lib}`)).items;
      list.replaceChildren(...(items.length ? items.map((item) => h("div", { class: "lib-item card" },
        item.has_thumb ? h("img", { class: "thumb", src: `/api/library/${lib}/${item.id}/thumb`, alt: "" }) : h("div", { class: "thumb" }),
        h("div", { class: "meta" }, h("b", {}, item.name),
          h("small", { class: "muted" }, [item.info?.width && `${item.info.width}×${item.info.height}`,
            item.info?.duration && fmtDuration(item.info.duration), fmtBytes(item.size)].filter(Boolean).join(" · "))),
        h("a", { class: "btn small", href: `/api/library/${lib}/${item.id}/file`, target: "_blank", rel: "noopener" }, "▶"),
        h("button", { class: "btn small", onclick: async () => {
          const name = prompt(t("ui.rename"), item.name);
          if (name) { await api(`/api/library/${lib}/${item.id}`, { method: "PATCH", body: { name } }); load(); }
        } }, "✎"),
        h("button", { class: "btn small danger", onclick: async () => {
          if (confirm(t("ui.confirm_delete", { name: item.name }))) { await api(`/api/library/${lib}/${item.id}`, { method: "DELETE" }); load(); }
        } }, "🗑"))) : [h("p", { class: "empty" }, t("ui.library_empty"))]));
    };
    await load();
  }
}

// ------------------------------------------------------------------ system
async function system(body) {
  const s = await api("/api/system");
  const kv = (rows) => h("dl", { class: "kv" }, ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v ?? "—")]));
  const mem = s.memory || {};
  body.replaceChildren(
    h("section", { class: "card" }, h("h2", { style: "margin-top:0" }, t("admin.device")), kv([
      [t("admin.platform"), s.platform], ["Python", s.python], [t("admin.version"), s.version],
      ["CPU", `${s.cpu_count} · load ${s.load ? s.load.map((x) => x.toFixed(2)).join(" ") : "—"}`],
      [t("admin.temperature"), s.temperature_c ? `${s.temperature_c.toFixed(1)} °C` : "—"],
      [t("admin.memory"), mem.MemTotal ? `${fmtBytes(mem.MemAvailable)} / ${fmtBytes(mem.MemTotal)}` : "—"],
      [t("admin.disk"), `${fmtBytes(s.disk.free)} / ${fmtBytes(s.disk.total)}`],
    ])),
    h("section", { class: "card" }, h("h2", { style: "margin-top:0" }, t("admin.storage")), kv([
      [t("admin.uploads"), fmtBytes(s.data.uploads)], [t("admin.outputs"), fmtBytes(s.data.outputs)],
      [t("admin.library"), fmtBytes(s.data.library)], [t("admin.models"), fmtBytes(s.data.models)],
    ]), h("div", { class: "row" },
      h("button", { class: "btn small", onclick: async () => { await api("/api/system/cleanup", { method: "POST", body: { uploads: true, tmp: true } }); toast(t("ui.done")); system(body); } }, t("admin.cleanup")))),
    h("section", { class: "card" }, h("h2", { style: "margin-top:0" }, t("admin.dependencies")), kv([
      ["ffmpeg", s.ffmpeg.available ? `${s.ffmpeg.version}${s.ffmpeg.zscale ? " · zscale (HDR)" : ""}${s.ffmpeg.hw_encoders.length ? ` · ${s.ffmpeg.hw_encoders.join(", ")}` : ""}` : "✕"],
      ["tesseract", s.tesseract.available ? s.tesseract.languages.join(", ") : "✕"],
      ...Object.entries(s.tools).map(([id, v]) => [t(`tool.${id}.name`, {}, id), v.available ? "✓" : `✕ ${t(`err.${v.reason}`, {}, v.reason)}`]),
    ])),
    h("section", { class: "card" }, h("h2", { style: "margin-top:0" }, t("admin.queue")), kv(
      Object.entries(s.jobs.lanes).map(([lane, v]) => [t(`lane.${lane}`, {}, lane), `${v.workers} ${t("admin.workers")} · ${v.waiting} ${t("admin.waiting")}`]))),
  );
}

// ------------------------------------------------------------------ security
async function security(body) {
  const data = await api("/api/settings");
  let password = "";
  body.replaceChildren(h("section", { class: "card stack" },
    h("h2", { style: "margin:0" }, t("admin.password")),
    h("p", { class: "muted", style: "margin:0" }, data.secrets.APP_PASSWORD.configured ? t("admin.password_on") : t("admin.password_off")),
    h("small", { class: "muted" }, t("admin.password_help")),
    h("input", { type: "password", autocomplete: "new-password", oninput: (e) => { password = e.target.value; } }),
    h("div", { class: "row" },
      h("button", { class: "btn primary", onclick: async () => {
        try { await api("/api/secrets", { method: "PUT", body: { APP_PASSWORD: password } }); toast(t("ui.saved")); security(body); }
        catch (e) { toast(errorText(e.error)); }
      } }, t("ui.save")),
      data.secrets.APP_PASSWORD.configured ? h("button", { class: "btn", onclick: async () => {
        await api("/api/logout", { method: "POST" }); location.reload();
      } }, t("admin.logout")) : null)));
}
