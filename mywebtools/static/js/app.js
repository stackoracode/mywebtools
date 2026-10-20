import {
  api, connectEvents, errorText, h, loadLanguage, onJobEvent, pickLanguage, refreshJobs, setUnauthorizedHandler,
  state, storage, t, toast, translateStatic,
} from "./core.js";
import { renderToolForm } from "./form.js";
import { jobList, renderJobs, updateBadge } from "./jobs.js";
import { enhanceAvatarForm, startEngineWatch } from "./avatar.js";
import { renderAdmin } from "./admin.js";

const view = document.getElementById("view");
const ICONS = {
  "file-text": "📄", layers: "📚", scissors: "✂️", image: "🖼️", "file-plus": "🗂️", minimize: "🗜️", "scan-text": "🔎",
  mic: "🎙️", music: "🎵", info: "ℹ️", "user-video": "🧑‍💻", tool: "🛠️",
};
const CATEGORY_ORDER = ["avatar", "video", "pdf", "media", "misc"];

// ---------------------------------------------------------------- views
function renderHome() {
  const cats = {};
  for (const tool of state.tools) (cats[tool.category] ||= []).push(tool);
  const order = [...CATEGORY_ORDER, ...Object.keys(cats).filter((c) => !CATEGORY_ORDER.includes(c))];
  view.replaceChildren(
    ...order.filter((c) => cats[c]).flatMap((cat) => [
      h("h2", { class: "cat-title" }, t(`cat.${cat}`, {}, cat)),
      h("div", { class: "grid" }, ...cats[cat].map((tool) => h("a", {
        class: `card tool-card${tool.available ? "" : " disabled"}`, href: `#/tool/${tool.id}` },
        h("span", { class: "ico", "aria-hidden": "true" }, ICONS[tool.icon] || ICONS.tool),
        h("div", {}, h("h3", {}, t(`tool.${tool.id}.name`, {}, tool.name)),
          h("p", {}, tool.available ? t(`tool.${tool.id}.desc`, {}, tool.description)
            : t(`err.${tool.unavailable_reason}`, {}, tool.unavailable_reason)))))),
    ]),
    h("h2", { class: "cat-title" }, t("home.recent")),
    jobList(() => true, 5),
  );
}

async function renderTool(id) {
  const tool = state.tools.find((x) => x.id === id);
  if (!tool) { view.replaceChildren(h("p", { class: "empty" }, t("err.tool.unknown", { tool: id }))); return; }
  const mount = h("div");
  const runs = h("div");
  view.replaceChildren(
    h("a", { href: "#/", class: "muted" }, `‹ ${t("nav.tools")}`),
    h("h1", {}, t(`tool.${id}.name`, {}, tool.name)),
    h("p", { class: "muted" }, t(`tool.${id}.desc`, {}, tool.description)),
    tool.available ? null : h("p", { class: "error" }, t(`err.${tool.unavailable_reason}`, {}, tool.unavailable_reason)),
    mount,
    h("h2", {}, t("tool.recent_runs")), runs);
  const hooks = { onSubmitted: () => runs.scrollIntoView({ behavior: "smooth" }) };
  const ctrl = await renderToolForm(tool, mount, hooks);
  if (id === "avatar_pip") enhanceAvatarForm(ctrl);
  runs.replaceChildren(jobList((j) => j.tool === id, 10));
}

// ---------------------------------------------------------------- router
async function route() {
  const [, page = "", arg = ""] = (location.hash || "#/").split("/");
  document.querySelectorAll(".tabbar a").forEach((a) => a.classList.toggle("active",
    a.dataset.nav === (page === "jobs" ? "jobs" : page === "admin" ? "admin" : "home")));
  try {
    if (page === "tool") await renderTool(decodeURIComponent(arg));
    else if (page === "jobs") renderJobs(view);
    else if (page === "admin") await renderAdmin(view, arg);
    else renderHome();
  } catch (e) {
    view.replaceChildren(h("p", { class: "error" }, errorText(e.error) || String(e)));
  }
  window.scrollTo(0, 0);
}

// ---------------------------------------------------------------- auth / theme / language
function showLogin() {
  const dlg = document.getElementById("login");
  if (dlg.open) return;
  dlg.showModal();
}

document.getElementById("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = document.getElementById("login-error");
  try {
    await api("/api/login", { method: "POST", body: { password: e.target.password.value } });
    location.reload();
  } catch (ex) { err.hidden = false; err.textContent = errorText(ex.error); }
});

function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme; else delete document.documentElement.dataset.theme;
}
document.getElementById("theme").addEventListener("click", () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const current = document.documentElement.dataset.theme || (dark ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
  storage.set("mwt.theme", next);
});

async function boot() {
  applyTheme(storage.get("mwt.theme"));
  setUnauthorizedHandler(showLogin);
  const auth = await api("/api/auth");
  state.meta = auth.ok || !auth.required ? await api("/api/meta") : { ui_languages: { en: "English", "zh-CN": "简体中文" } };
  const langSelect = document.getElementById("lang");
  langSelect.replaceChildren(...Object.entries(state.meta.ui_languages).map(([code, name]) => h("option", { value: code }, name)));
  await loadLanguage(pickLanguage(state.meta.ui_languages));
  langSelect.value = state.lang;
  langSelect.addEventListener("change", async () => { await loadLanguage(langSelect.value); route(); });
  if (auth.required && !auth.ok) { showLogin(); return; }

  state.tools = await api("/api/tools");
  await refreshJobs().catch(() => {});
  connectEvents();
  const announced = new Set();
  onJobEvent((event) => {
    updateBadge();
    if (event.type === "job" && event.job.state === "review" && !announced.has(event.job.id)) {
      announced.add(event.job.id);
      toast(t("review.ready", { title: event.job.title }), 6000);
    }
    if (event.type === "job" && ["done", "failed"].includes(event.job.state) && event.job.finished
        && Date.now() / 1000 - event.job.finished < 5) {
      toast(`${event.job.title}: ${t(`state.${event.job.state}`)}`);
    }
  });
  updateBadge();
  startEngineWatch();
  setInterval(() => { if (document.visibilityState === "visible") refreshJobs().catch(() => {}); }, 30000);
  window.addEventListener("hashchange", route);
  translateStatic();
  route();
}

boot().catch((e) => { view.replaceChildren(h("p", { class: "error" }, errorText(e.error) || String(e))); });
