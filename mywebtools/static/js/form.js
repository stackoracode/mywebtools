// Renders a tool form from the server-side schema, with memory (last used values +
// unsent draft), presets, conditional fields and background uploads.

import { api, errorText, fmtBytes, h, languageName, state, storage, t, toast, uploadFile } from "./core.js";

const LANGUAGE_PARAMS = new Set(["ocr_langs", "language"]);

export async function renderToolForm(tool, mount, hooks = {}) {
  const tid = tool.id;
  const L = (name, fb) => t(`tool.${tid}.p.${name}`, {}, fb);
  const draftKey = `mwt.draft.${tid}`;
  const prefs = await api(`/api/prefs/${tid}`).catch(() => ({ last: {}, presets: {} }));
  let presets = prefs.presets || {};

  const params = tool.params;
  const byName = Object.fromEntries(params.map((p) => [p.name, p]));
  const values = {};
  for (const p of params) if (!["file", "info"].includes(p.type)) values[p.name] = clone(p.default);
  const known = (obj) => Object.fromEntries(Object.entries(obj || {}).filter(([k]) => k in values));
  Object.assign(values, known(prefs.last), known(storage.get(draftKey, {})));

  const files = {}; // name -> [{id, name, size, progress, status, file}]
  const fields = {}; // name -> {wrap, set}
  let showAdvanced = storage.get(`mwt.adv.${tid}`, false);

  const form = h("form", { class: "tool-form", novalidate: true });

  // ---- presets ----
  const presetSelect = h("select", { "aria-label": t("ui.presets") });
  const fillPresets = () => {
    presetSelect.replaceChildren(h("option", { value: "" }, t("ui.presets_choose")),
      ...Object.keys(presets).sort().map((n) => h("option", { value: n }, n)));
  };
  fillPresets();
  presetSelect.addEventListener("change", () => {
    const chosen = presets[presetSelect.value];
    if (chosen) { setAll({ ...defaults(), ...known(chosen) }); toast(t("ui.preset_loaded", { name: presetSelect.value })); }
  });
  const presetBar = h("div", { class: "card row" },
    h("div", { class: "spacer", style: "min-width:160px" }, presetSelect),
    h("button", { type: "button", class: "btn small", onclick: async () => {
      const name = prompt(t("ui.preset_name"), presetSelect.value || "");
      if (!name) return;
      try {
        presets = (await api(`/api/prefs/${tid}/presets/${encodeURIComponent(name)}`, { method: "PUT", body: { values } })).presets;
        fillPresets(); presetSelect.value = name; toast(t("ui.preset_saved", { name }));
      } catch (e) { toast(errorText(e.error)); }
    } }, t("ui.save")),
    h("button", { type: "button", class: "btn small danger", onclick: async () => {
      if (!presetSelect.value || !confirm(t("ui.confirm_delete", { name: presetSelect.value }))) return;
      presets = (await api(`/api/prefs/${tid}/presets/${encodeURIComponent(presetSelect.value)}`, { method: "DELETE" })).presets;
      fillPresets();
    } }, t("ui.delete")),
    h("button", { type: "button", class: "btn small", onclick: () => { setAll(defaults()); toast(t("ui.reset_done")); } }, t("ui.reset")),
  );
  form.append(presetBar);

  // ---- groups ----
  const groups = tool.groups?.length ? tool.groups : [{ name: "main", label: "" }];
  const bodies = {};
  groups.forEach((g, i) => {
    const body = h("div", { class: "group-body" });
    bodies[g.name] = body;
    const label = t(`tool.${tid}.g.${g.name}`, {}, g.label || t("ui.settings"));
    const details = h("details", { class: "group", open: !g.collapsed || i === 0 }, h("summary", {}, label), body);
    details.dataset.group = g.name;
    form.append(details);
  });

  for (const p of params) {
    const field = buildField(p);
    fields[p.name] = field;
    (bodies[p.group] || bodies[groups[0].name]).append(field.wrap);
  }

  const advToggle = h("label", { class: "switch card" }, h("span", {}, t("ui.show_advanced")),
    h("input", { type: "checkbox", checked: showAdvanced, onchange: (e) => {
      showAdvanced = e.target.checked; storage.set(`mwt.adv.${tid}`, showAdvanced); refresh();
    } }));
  if (params.some((p) => p.advanced)) form.append(advToggle);

  const submitBtn = h("button", { type: "submit", class: "btn primary" }, t("ui.run"));
  form.append(h("div", { class: "submit-bar" }, submitBtn));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (Object.values(files).flat().some((f) => f.status === "uploading")) return toast(t("ui.wait_uploads"));
    for (const p of params) {
      if (p.type === "file" && p.required && visible(p) && !(files[p.name] || []).some((f) => f.id)) {
        fields[p.name].wrap.scrollIntoView({ behavior: "smooth", block: "center" });
        return toast(t("err.input.missing", { field: L(p.name, p.label) }));
      }
    }
    if (hooks.beforeSubmit && !(await hooks.beforeSubmit())) return;
    submitBtn.disabled = true;
    try {
      const data = await api("/api/jobs", { method: "POST", body: { tool: tid, params: values, files: fileIds() } });
      storage.remove(draftKey);
      toast(t("ui.queued", { n: data.jobs.length }));
      hooks.onSubmitted?.(data.jobs);
    } catch (err) { toast(errorText(err.error)); } finally { submitBtn.disabled = false; }
  });

  mount.replaceChildren(form);
  refresh();

  // ---------- field builders ----------
  function buildField(p) {
    const id = `f-${tid}-${p.name}`;
    const wrap = h("div", { class: `field${p.advanced ? " adv" : ""}${["file", "textarea", "info", "library"].includes(p.type) || p.type === "multiselect" ? " wide" : ""}` });
    const label = L(p.name, p.label);
    const help = p.help ? h("div", { class: "help" }, t(`tool.${tid}.h.${p.name}`, {}, p.help)) : null;
    const unit = p.unit ? t(`unit.${p.unit}`, {}, p.unit) : "";
    let set = () => {};

    if (p.type === "bool") {
      const input = h("input", { type: "checkbox", id, onchange: (e) => change(p.name, e.target.checked) });
      wrap.append(h("label", { class: "switch", for: id }, h("span", {}, label), input), help);
      set = (v) => { input.checked = !!v; };
    } else if (p.type === "select") {
      const select = h("select", { id, onchange: (e) => change(p.name, e.target.value) },
        ...(p.options || []).map(([v, lab]) => h("option", { value: v }, optionLabel(p, v, lab))));
      wrap.append(h("label", { for: id }, label), select, help);
      set = (v) => { select.value = v ?? ""; };
    } else if (p.type === "multiselect") {
      const chips = h("div", { class: "chips", role: "group", "aria-label": label });
      for (const [v, lab] of p.options || []) {
        chips.append(h("button", { type: "button", class: "chip", "aria-pressed": "false", dataset: { v }, onclick: (e) => {
          const list = new Set(values[p.name] || []);
          list.has(v) ? list.delete(v) : list.add(v);
          const ordered = (p.options || []).map((o) => o[0]).filter((x) => list.has(x));
          change(p.name, ordered);
          e.currentTarget.setAttribute("aria-pressed", String(list.has(v)));
        } }, optionLabel(p, v, lab)));
      }
      if (!(p.options || []).length) chips.append(h("span", { class: "muted" }, t("ui.none_installed")));
      wrap.append(h("div", { class: "label" }, label), chips, help);
      set = (v) => chips.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String((v || []).includes(c.dataset.v))));
    } else if (p.type === "number" || p.type === "int") {
      const input = h("input", { type: "number", id, inputmode: p.type === "int" ? "numeric" : "decimal",
        min: p.min, max: p.max, step: p.step ?? (p.type === "int" ? 1 : "any"),
        onchange: (e) => change(p.name, e.target.value === "" ? p.default : Number(e.target.value)) });
      wrap.append(h("label", { for: id }, label), h("div", { class: "with-unit" }, input, unit && h("span", {}, unit)), help);
      set = (v) => { input.value = v ?? ""; };
    } else if (p.type === "range") {
      const out = h("output", { for: id });
      const input = h("input", { type: "range", id, min: p.min ?? 0, max: p.max ?? 100, step: p.step ?? 1,
        oninput: (e) => { out.textContent = e.target.value + (unit ? ` ${unit}` : ""); change(p.name, Number(e.target.value)); } });
      wrap.append(h("label", { for: id, class: "row" }, h("span", { class: "spacer" }, label), out), input, help);
      set = (v) => { input.value = v; out.textContent = v + (unit ? ` ${unit}` : ""); };
    } else if (p.type === "color") {
      const input = h("input", { type: "color", id, oninput: (e) => change(p.name, e.target.value) });
      wrap.append(h("label", { for: id }, label), input, help);
      set = (v) => { input.value = v || "#000000"; };
    } else if (p.type === "textarea") {
      const input = h("textarea", { id, oninput: (e) => change(p.name, e.target.value) });
      wrap.append(h("label", { for: id }, label), input, help);
      set = (v) => { input.value = v ?? ""; };
    } else if (p.type === "file") {
      buildFile(p, wrap, label, help, id);
    } else if (p.type === "library") {
      set = buildLibrary(p, wrap, label, help, id);
    } else if (p.type === "info") {
      wrap.append(h("div", { class: "info-slot", dataset: { info: p.name } }));
    } else {
      const input = h("input", { type: "text", id, oninput: (e) => change(p.name, e.target.value) });
      wrap.append(h("label", { for: id }, label), input, help);
      set = (v) => { input.value = v ?? ""; };
    }
    if (p.name in values) set(values[p.name]);
    return { wrap, set };
  }

  function optionLabel(p, value, english) {
    if (LANGUAGE_PARAMS.has(p.name) && value !== "auto") return languageName(value);
    return t(`tool.${tid}.o.${p.name}.${value}`, {}, english);
  }

  function buildFile(p, wrap, label, help, id) {
    const list = h("ul", { class: "files" });
    const input = h("input", { type: "file", id, accept: p.accept || undefined, multiple: p.multiple || undefined,
      onchange: (e) => { addFiles([...e.target.files]); e.target.value = ""; } });
    const drop = h("label", { class: "drop", for: id }, input,
      h("span", {}, p.multiple ? t("ui.pick_files") : t("ui.pick_file")));
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); addFiles([...e.dataTransfer.files]); });
    wrap.append(h("div", { class: "label" }, label), drop, list, help);
    files[p.name] = [];

    function draw() {
      list.replaceChildren(...files[p.name].map((f) => h("li", {},
        h("span", { class: "name", title: f.name }, f.name),
        h("small", {}, fmtBytes(f.size)),
        f.status === "uploading" ? h("progress", { max: 1, value: f.progress || 0 }) :
          f.status === "error" ? h("span", { class: "bad", title: f.error }, "✕") : h("span", { class: "ok" }, "✓"),
        h("button", { type: "button", class: "icon-btn", "aria-label": t("ui.remove"), onclick: () => {
          f.file?._xhr?.abort();
          files[p.name] = files[p.name].filter((x) => x !== f);
          draw(); hooks.onFilesChanged?.(p.name, files[p.name]);
        } }, "×"))));
    }

    function addFiles(chosen) {
      if (!chosen.length) return;
      if (!p.multiple) { files[p.name].forEach((f) => f.file?._xhr?.abort()); files[p.name] = []; chosen = chosen.slice(0, 1); }
      for (const file of chosen) {
        const entry = { name: file.name, size: file.size, status: "uploading", progress: 0, file };
        files[p.name].push(entry);
        uploadFile(file, (frac) => { entry.progress = frac; draw(); })
          .then((data) => { entry.id = data.id; entry.status = "done"; draw(); hooks.onFilesChanged?.(p.name, files[p.name]); })
          .catch((err) => { entry.status = "error"; entry.error = errorText(err.error); draw(); toast(entry.error); });
      }
      draw();
    }
  }

  function buildLibrary(p, wrap, label, help, id) {
    const select = h("select", { id, onchange: (e) => { change(p.name, e.target.value); showThumb(); } });
    const thumb = h("img", { class: "thumb", alt: "", hidden: true });
    const picker = h("input", { type: "file", accept: state.meta?.libraries?.[p.library]?.accept || "video/*", hidden: true,
      onchange: async (e) => {
        const file = e.target.files[0];
        e.target.value = "";
        if (!file) return;
        addBtn.disabled = true;
        try {
          const up = await uploadFile(file, (f) => { addBtn.textContent = `${Math.round(f * 100)}%`; });
          addBtn.textContent = "…";
          const item = await api(`/api/library/${p.library}`, { method: "POST", body: { upload_id: up.id, name: file.name.replace(/\.[^.]+$/, "") } });
          await load(item.id);
          toast(t("ui.added_to_library"));
        } catch (err) { toast(errorText(err.error)); } finally { addBtn.disabled = false; addBtn.textContent = t("ui.upload_new"); }
      } });
    const addBtn = h("button", { type: "button", class: "btn small", onclick: () => picker.click() }, t("ui.upload_new"));
    wrap.append(h("label", { for: id }, label), h("div", { class: "lib-row" }, thumb, select, addBtn, picker), help);
    let items = [];

    function showThumb() {
      const item = items.find((i) => i.id === select.value);
      thumb.hidden = !item?.has_thumb;
      if (item?.has_thumb) thumb.src = `/api/library/${p.library}/${item.id}/thumb`;
      hooks.onLibraryChanged?.(p.name, item);
    }
    async function load(selectId) {
      items = (await api(`/api/library/${p.library}`).catch(() => ({ items: [] }))).items;
      select.replaceChildren(h("option", { value: "" }, items.length ? t("ui.choose") : t("ui.library_empty")),
        ...items.map((i) => h("option", { value: i.id }, `${i.name}${i.info?.duration ? ` · ${Math.round(i.info.duration)}s` : ""}`)));
      if (selectId) change(p.name, selectId);
      select.value = items.some((i) => i.id === values[p.name]) ? values[p.name] : "";
      if (select.value !== values[p.name]) values[p.name] = select.value;
      showThumb();
    }
    load();
    return (v) => { select.value = v || ""; showThumb(); };
  }

  // ---------- state ----------
  function change(name, value) {
    values[name] = value;
    storage.set(draftKey, values);
    refresh();
    hooks.onChange?.(name, values);
  }

  function visible(p) {
    if (p.advanced && !showAdvanced) return false;
    for (const [key, expected] of Object.entries(p.show_if || {})) {
      const allowed = Array.isArray(expected) ? expected : [expected];
      if (key in byName && !visible(byName[key])) return false;
      if (!allowed.includes(values[key])) return false;
    }
    return true;
  }

  function refresh() {
    for (const p of params) fields[p.name].wrap.hidden = !visible(p);
    form.querySelectorAll("details.group").forEach((d) => {
      d.hidden = ![...d.querySelectorAll(".group-body > .field")].some((f) => !f.hidden);
    });
  }

  function defaults() {
    const out = {};
    for (const p of params) if (p.name in values) out[p.name] = clone(p.default);
    return out;
  }

  function setAll(next) {
    Object.assign(values, next);
    for (const [name, v] of Object.entries(next)) fields[name]?.set(v);
    storage.set(draftKey, values);
    refresh();
    hooks.onChange?.("*", values);
  }

  function fileIds() {
    const out = {};
    for (const [name, list] of Object.entries(files)) {
      if (visible(byName[name])) out[name] = list.filter((f) => f.id).map((f) => f.id);
    }
    return out;
  }

  return { values, files, fileIds, form, refresh, setAll, hooks, slot: (name) => form.querySelector(`[data-info="${name}"]`) };
}

function clone(v) { return v && typeof v === "object" ? JSON.parse(JSON.stringify(v)) : v; }
