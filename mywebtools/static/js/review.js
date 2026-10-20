// Subtitle draft review: edit lines/timing, ask the AI to revise, approve to continue the job.
import { api, errorText, h, t, toast } from "./core.js";

const fmt = (sec) => {
  const ms = Math.round(Math.max(sec, 0) * 1000);
  const m = Math.floor(ms / 60000), s = Math.floor((ms % 60000) / 1000), r = ms % 1000;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(r).padStart(3, "0")}`;
};
const parse = (text) => {
  const m = String(text).trim().match(/^(?:(\d+):)?(\d+)(?:[.,](\d{1,3}))?$/);
  if (!m) return NaN;
  return (Number(m[1] || 0) * 60) + Number(m[2]) + Number((m[3] || "0").padEnd(3, "0")) / 1000;
};
const toSrt = (cues) => cues.map((c, i) => `${i + 1}\n${srtTime(c.start)} --> ${srtTime(c.end)}\n${c.text}\n`).join("\n");
const srtTime = (sec) => { const [mm, rest] = fmt(sec).split(":"); const h = Math.floor(Number(mm) / 60); return `${String(h).padStart(2, "0")}:${String(Number(mm) % 60).padStart(2, "0")}:${rest.replace(".", ",")}`; };
function fromSrt(text) {
  const out = [];
  for (const block of text.replace(/\r/g, "").trim().split(/\n\s*\n/)) {
    const lines = block.split("\n");
    const i = lines.findIndex((l) => l.includes("-->"));
    if (i < 0) continue;
    const [a, b] = lines[i].split("-->").map((x) => x.trim().replace(",", "."));
    const toSec = (v) => { const p = v.split(":").map(Number); return p.length === 3 ? p[0] * 3600 + p[1] * 60 + p[2] : p[0] * 60 + p[1]; };
    const body = lines.slice(i + 1).join("\n").trim();
    if (body) out.push({ start: toSec(a), end: toSec(b), text: body });
  }
  return out;
}

export async function openReview(job) {
  let data;
  try { data = await api(`/api/jobs/${job.id}/review`); } catch (e) { return toast(errorText(e.error)); }
  let cues = (data.draft?.cues || []).map((c) => ({ ...c }));
  const draft = data.draft || {};
  let srtMode = false;

  const list = h("div", { class: "review-list" });
  const srtArea = h("textarea", { class: "review-srt", hidden: true, spellcheck: "false" });
  const instruction = h("input", { type: "text", placeholder: t("review.instruction_ph") });
  const target = h("select", {}, ...Object.entries({ en: "English", "zh-Hans": "简体中文", "zh-Hant": "繁體中文", ja: "日本語", ko: "한국어",
    es: "Español", fr: "Français", de: "Deutsch", pt: "Português", ru: "Русский" }).map(([k, v]) => h("option", { value: k, selected: k === draft.target }, v)));
  const busy = (on) => dlg.querySelectorAll("button").forEach((b) => { b.disabled = on; });

  const sync = () => { if (srtMode) cues = fromSrt(srtArea.value); };
  const draw = () => {
    list.replaceChildren(...cues.map((c, i) => {
      const start = h("input", { type: "text", inputmode: "decimal", value: fmt(c.start), "aria-label": t("review.start"),
        onchange: (e) => { const v = parse(e.target.value); if (!Number.isNaN(v)) c.start = v; e.target.value = fmt(c.start); } });
      const end = h("input", { type: "text", inputmode: "decimal", value: fmt(c.end), "aria-label": t("review.end"),
        onchange: (e) => { const v = parse(e.target.value); if (!Number.isNaN(v)) c.end = v; e.target.value = fmt(c.end); } });
      const text = h("textarea", { rows: Math.min(4, (c.text.match(/\n/g) || []).length + 1), oninput: (e) => { c.text = e.target.value; } });
      text.value = c.text;
      return h("div", { class: "review-row" },
        h("div", { class: "review-times" }, h("small", { class: "muted" }, `#${i + 1}`), start, end),
        text,
        h("div", { class: "review-actions" },
          h("button", { type: "button", class: "icon-btn", title: t("review.insert"), onclick: () => {
            const next = cues[i + 1];
            cues.splice(i + 1, 0, { start: c.end, end: next ? Math.max(c.end + 0.5, next.start) : c.end + 2, text: "" });
            draw();
          } }, "＋"),
          i < cues.length - 1 ? h("button", { type: "button", class: "icon-btn", title: t("review.merge"), onclick: () => {
            const next = cues[i + 1];
            c.end = next.end; c.text = `${c.text} ${next.text}`.trim(); cues.splice(i + 1, 1); draw();
          } }, "⤵") : null,
          h("button", { type: "button", class: "icon-btn", title: t("ui.delete"), onclick: () => { cues.splice(i, 1); draw(); } }, "×")));
    }));
  };

  async function revise(mode) {
    sync();
    busy(true);
    try {
      const res = await api(`/api/jobs/${job.id}/review/revise`, { method: "POST",
        body: { cues, mode, target: target.value, instruction: instruction.value } });
      cues = res.cues;
      srtArea.value = toSrt(cues);
      draw();
      toast(t("review.revised"));
    } catch (e) { toast(errorText(e.error)); } finally { busy(false); }
  }

  const dlg = h("dialog", { class: "sheet review" },
    h("form", { method: "dialog", onsubmit: (e) => e.preventDefault() },
      h("h2", {}, t("review.title")),
      h("p", { class: "muted", style: "margin:0" }, t("review.intro", { n: cues.length })),
      h("div", { class: "row" },
        h("button", { type: "button", class: "btn small", onclick: () => {
          if (srtMode) { cues = fromSrt(srtArea.value); draw(); } else srtArea.value = toSrt(cues);
          srtMode = !srtMode; srtArea.hidden = !srtMode; list.hidden = srtMode;
        } }, t("review.toggle_srt"))),
      list, srtArea,
      h("details", { class: "card" }, h("summary", {}, t("review.ai")),
        h("div", { class: "stack", style: "margin-top:8px" },
          h("div", { class: "row" },
            h("button", { type: "button", class: "btn small", onclick: () => revise("fix") }, t("review.ai_fix")),
            target,
            h("button", { type: "button", class: "btn small", onclick: () => revise("translate") }, t("review.ai_translate")),
            h("button", { type: "button", class: "btn small", onclick: () => revise("bilingual") }, t("review.ai_bilingual"))),
          h("div", { class: "row" }, h("div", { class: "spacer" }, instruction),
            h("button", { type: "button", class: "btn small", onclick: () => { if (instruction.value.trim()) revise("revise"); } },
              t("review.ai_apply"))))),
      h("div", { class: "row" },
        h("button", { type: "button", class: "btn", onclick: () => dlg.close() }, t("review.later")),
        h("span", { class: "spacer" }),
        h("button", { type: "button", class: "btn primary", onclick: async () => {
          sync();
          const clean = cues.filter((c) => c.text.trim() && c.end > c.start);
          if (!clean.length) return toast(t("err.subs.empty"));
          busy(true);
          try {
            await api(`/api/jobs/${job.id}/review`, { method: "POST", body: { cues: clean } });
            toast(t("review.approved"));
            dlg.close();
          } catch (e) { toast(errorText(e.error)); busy(false); }
        } }, t("review.approve")))));
  document.body.append(dlg);
  dlg.addEventListener("close", () => dlg.remove());
  draw();
  dlg.showModal();
}
