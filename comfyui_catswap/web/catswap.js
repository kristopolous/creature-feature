// Replaces App Mode's stock empty state ("A simplified view that hides the node graph...") with what this App
// does and how to use it, only when the open workflow is a Creature Feature one.
import { app } from "../../scripts/app.js";

const ANIMALS = ["cat", "dog", "parrot", "goose", "camel", "llama", "capybara", "penguin", "goat", "raccoon",
                 "corgi", "owl", "alpaca", "pig", "rabbit", "duck", "hedgehog"];

function ours() {
  return (app.graph?._nodes || []).some((n) => String(n.type || n.comfyClass).startsWith("CatSwap"));
}

function hero() {
  const d = document.createElement("div");
  d.className = "catswap-hero";
  d.style.cssText = "max-width:34rem;margin:0 auto;text-align:center;line-height:1.5";
  const chips = ANIMALS.map((a) =>
    `<span style="display:inline-block;margin:3px;padding:2px 10px;border-radius:999px;` +
    `border:1px solid var(--border-color,#444);font-size:.85rem">${a}</span>`).join("");
  d.innerHTML = `
    <h2 style="font-size:1.8rem;font-weight:600;margin:0 0 .5rem">Creature Feature</h2>
    <p style="opacity:.85;margin:0 0 1rem">Turn one person in a movie scene into an animal. The animal acts like an
      animal; everyone else carries on the scene as if nothing is wrong.</p>
    <ol style="text-align:left;opacity:.85;margin:0 auto 1rem;padding-left:1.2rem;max-width:30rem">
      <li>Pick a <b>starter</b> scene, or choose <i>my video</i> and upload one.</li>
      <li>Optionally <b>trim</b> to the part you want (the first 60 s by default).</li>
      <li>Leave <b>swaps</b> empty for the scene's suggestion, or say who becomes the animal:<br>
        <code style="opacity:.9">the young man in the gray hoodie = cat</code><br>
        <span style="font-size:.85rem;opacity:.75">Describe what you can see (clothes, hair), not the character's name.</span></li>
      <li>Press <b>Run</b>. A decision model finds that person in every shot; each one is swapped and the scene is
        put back together over the original audio.</li>
    </ol>
    <div style="opacity:.8;font-size:.85rem;margin-bottom:.3rem">Animal ideas (anything works)</div>
    <div>${chips}</div>`;
  return d;
}

function apply() {
  const show = ours();
  for (const el of document.querySelectorAll('[data-testid="linear-welcome"]')) {
    const mine = el.querySelector(":scope > .catswap-hero");
    for (const c of el.children) if (c !== mine) c.style.display = show ? "none" : "";
    if (show && !mine) el.appendChild(hero());
    if (!show && mine) mine.remove();
  }
}

app.registerExtension({
  name: "catswap.welcome",
  setup() {
    new MutationObserver(apply).observe(document.body, { childList: true, subtree: true });
    apply();
  },
});
