import { api, errorText, fmtBytes, fmtDuration, fmtTime, h, onJobEvent, state, t, toast, toolName } from "./core.js";

const VIDEO = /\.(mp4|mov|webm|m4v)$/i;
const IMAGE = /\.(png|jpe?g|webp)$/i;

export function jobCard(job) {
  const card = h("article", { class: "card job", dataset: { id: job.id } });
  fillCard(card, job);
  return card;
}

function fillCard(card, job) {
  const active = job.state === "queued" || job.state === "running";
  const pct = job.progress ?? 0;
  const stage = job.stage ? t(`stage.${job.stage}`, job.stage_params, job.stage) : "";
  const elapsed = job.started ? ((job.finished || Date.now() / 1000) - job.started) : 0;
  const files = (job.outputs || []).map((o) => {
    const url = `/api/jobs/${job.id}/files/${encodeURIComponent(o.name)}`;
    return h("a", { class: "btn small", href: url, download: o.name }, `⬇ ${o.name}`, h("small", {}, ` ${fmtBytes(o.size)}`));
  });
  const media = (job.outputs || []).find((o) => VIDEO.test(o.name) || IMAGE.test(o.name));
  let viewer = null;
  if (media && job.state === "done") {
    const url = `/api/jobs/${job.id}/files/${encodeURIComponent(media.name)}?inline=1`;
    viewer = h("details", {}, h("summary", { class: "btn small" }, t("ui.preview")),
      VIDEO.test(media.name) ? h("video", { src: url, controls: true, playsinline: true, preload: "none" }) :
        h("img", { class: "preview", src: url, alt: media.name, loading: "lazy" }));
  }
  const result = job.result && Object.keys(job.result).length
    ? h("dl", { class: "kv" }, ...Object.entries(job.result).flatMap(([k, v]) => [
      h("dt", {}, t(`result.${k}`, {}, k)), h("dd", {}, formatValue(k, v))]))
    : null;

  card.replaceChildren(
    h("div", { class: "job-head" },
      h("span", { class: "job-title", title: job.title }, job.title || toolName(job.tool)),
      h("span", { class: `state ${job.state}` }, t(`state.${job.state}`, {}, job.state))),
    h("small", { class: "muted" }, `${toolName(job.tool)} · ${fmtTime(job.created)}${elapsed ? ` · ${fmtDuration(elapsed)}` : ""}`),
    active ? h("div", { class: `bar${job.state === "queued" || job.progress === null ? " indet" : ""}` }, h("i", { style: `width:${pct}%` })) : null,
    active && (stage || job.state === "queued") ? h("small", {}, job.state === "queued" ? t("state.waiting") : `${stage} · ${Math.round(pct)}%`) : null,
    job.error ? h("p", { class: "error" }, errorText(job.error)) : null,
    result,
    files.length ? h("div", { class: "outputs" }, files) : null,
    viewer,
    h("div", { class: "row" },
      active ? h("button", { class: "btn small danger", onclick: () => act(job, "cancel") }, t("ui.cancel")) : null,
      ["failed", "cancelled", "interrupted", "done"].includes(job.state)
        ? h("button", { class: "btn small", onclick: () => act(job, "retry") },
          job.meta?.resume_duix_task && job.state !== "done" ? t("ui.resume") : t("ui.retry")) : null,
      !active ? h("button", { class: "btn small", onclick: () => showLog(job) }, t("ui.log")) : null,
      h("span", { class: "spacer" }),
      !active ? h("button", { class: "btn small danger", onclick: () => act(job, "delete") }, t("ui.delete")) : null),
  );
}

function formatValue(key, value) {
  if (value === null || value === undefined) return "";
  if (["before", "after", "size"].includes(key)) return fmtBytes(value);
  if (Array.isArray(value)) return value.map((v) => t(`note.${v}`, {}, v)).join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "✓" : "✗";
  return String(value);
}

async function act(job, action) {
  try {
    if (action === "delete") {
      if (!confirm(t("ui.confirm_delete", { name: job.title }))) return;
      await api(`/api/jobs/${job.id}`, { method: "DELETE" });
      state.jobs.delete(job.id);
      document.querySelectorAll(`.job[data-id="${job.id}"]`).forEach((el) => el.remove());
    } else {
      await api(`/api/jobs/${job.id}/${action}`, { method: "POST" });
      toast(t(`ui.${action}_done`));
    }
  } catch (e) { toast(errorText(e.error)); }
}

async function showLog(job) {
  const full = await api(`/api/jobs/${job.id}`).catch(() => null);
  const dlg = h("dialog", { class: "sheet" },
    h("form", { method: "dialog" },
      h("h2", {}, job.title),
      h("dl", { class: "kv" }, ...Object.entries(full?.params || {}).flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, formatValue(k, v))])),
      h("pre", { class: "log" }, (full?.logs || []).join("\n") || "—"),
      h("button", { class: "btn block" }, t("ui.close"))));
  document.body.append(dlg);
  dlg.addEventListener("close", () => dlg.remove());
  dlg.showModal();
}

/** Live list of jobs (optionally filtered) that keeps itself up to date. */
export function jobList(filter = () => true, limit = 50) {
  const list = h("div", { class: "jobs" });
  const draw = () => {
    const jobs = [...state.jobs.values()].filter(filter).sort((a, b) => b.created - a.created).slice(0, limit);
    list.replaceChildren(...(jobs.length ? jobs.map(jobCard) : [h("p", { class: "empty" }, t("ui.no_jobs"))]));
  };
  draw();
  const off = onJobEvent((event) => {
    if (!list.isConnected) return off();
    if (event.type === "job" && filter(event.job)) {
      const card = list.querySelector(`.job[data-id="${event.job.id}"]`);
      card ? fillCard(card, event.job) : draw();
    } else if (event.type !== "job") draw();
  });
  return list;
}

export function renderJobs(view) {
  let filter = "all";
  const filters = ["all", "active", "done", "failed"];
  const tabs = h("div", { class: "tabs", role: "tablist" });
  const holder = h("div");
  const match = (j) => filter === "all" || (filter === "active" ? ["queued", "running"].includes(j.state)
    : filter === "failed" ? ["failed", "cancelled", "interrupted"].includes(j.state) : j.state === filter);
  const draw = () => {
    tabs.replaceChildren(...filters.map((f) => h("button", { class: "btn small", role: "tab", "aria-selected": String(f === filter),
      onclick: () => { filter = f; draw(); } }, t(`jobs.filter.${f}`, {}, f))));
    holder.replaceChildren(jobList(match, 200));
  };
  view.replaceChildren(h("h1", {}, t("nav.jobs")), tabs, holder,
    h("div", { class: "row", style: "margin-top:16px" }, h("button", { class: "btn small danger", onclick: async () => {
      if (!confirm(t("jobs.clear_confirm"))) return;
      await api("/api/jobs/clear", { method: "POST" });
      [...state.jobs.values()].filter((j) => !["queued", "running"].includes(j.state)).forEach((j) => state.jobs.delete(j.id));
      draw();
    } }, t("jobs.clear"))));
  draw();
}

export function updateBadge() {
  const n = [...state.jobs.values()].filter((j) => j.state === "queued" || j.state === "running").length;
  const badge = document.getElementById("jobs-badge");
  badge.hidden = !n;
  badge.textContent = n;
}
