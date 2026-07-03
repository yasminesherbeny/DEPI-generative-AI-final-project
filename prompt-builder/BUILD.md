# Nabra Prompt Builder — How It Was Built

A static, single-page web tool for constructing prompts for the project's GPT-2 + LoRA
product description generator ([`src/inference/generate.py`](../src/inference/generate.py)).
No build step, no dependencies — plain HTML, CSS, and JavaScript served as static files.

## Why a prompt builder

The fine-tuned model expects input ending in a `Description:` marker, which
`evaluate_model.py` uses to split the prompt from the generated continuation:

```
Product: <name>
Color: <color>
Description:
```

Constructing that text by hand for every product is tedious and error-prone, especially
once you want to steer tone, length, and style. The prompt builder turns a form into a
correctly-terminated prompt, and also emits ready-to-run Python that calls
`load_generator()` / `generate_text()`-style inference with the chosen sampling
parameters (temperature, top-k, top-p, max length, number of variations).

## Architecture

Three files, no framework:

| File | Responsibility |
|---|---|
| `index.html` | Semantic structure: form fields, chip groups, sliders, preview pane, history sidebar |
| `style.css` | All visual design — dark/light theme via `data-theme`, glassmorphism cards, chip/toggle states, responsive layout |
| `script.js` | All behavior — an IIFE with no external dependencies |

`script.js` is organized into focused sections (each a group of functions, not classes —
this is a small enough surface that classes would just add ceremony):

- **Prompt assembly** (`buildPrompt`, `updatePreview`) — reads every field/param and
  renders the live preview + character count on every input event.
- **Templates** — six presets (`ecommerce`, `social`, `technical`, `story`, `luxury`,
  `budget`) that set tone/length/style/CTA in one click. Picking a template highlights
  its card via a `.template-card--active` class (added to `style.css` — it existed for
  every other interactive control but had been missed for templates).
- **Chip groups** — a single generic handler (`setupChipGroup`) drives the tone, length,
  and style radio-button-style groups instead of three near-duplicate listeners.
- **Advanced parameters** — five range sliders bound generically via a config array
  (`[inputId, labelId, parser]`) so adding a sixth parameter later is a one-line change.
- **Actions** — copy prompt, copy as Python, download JSON, save to history, reset.
  Clipboard writes use the async Clipboard API with a `document.execCommand('copy')`
  fallback for browsers/contexts where it's unavailable.
- **History** — persisted to `localStorage` (`nabra.promptHistory`), rendered into the
  slide-out sidebar with per-item load/delete, plus export-all and clear-all.
- **Theme** — persisted to `localStorage` (`nabra.theme`), toggles the `data-theme`
  attribute the CSS keys off of.
- **Particle background** — a lightweight canvas animation (particle count scaled to
  viewport area, capped at 70) purely for visual polish; it doesn't affect app state.

State lives in two plain objects (`state` for tone/length/style/CTA/active template,
`params` for generation hyperparameters) rather than a framework store — the DOM is the
source of truth for form fields, and these objects just track the "selected chip" state
that isn't naturally queryable from a single input element.

## Generated prompt format

```
Generate a product description with the following specifications:

Product Name: AirPods Pro 2
Brand: Apple
Category: Electronics
Target Audience: Tech-savvy professionals aged 25-40
Key Features: Active Noise Cancellation, Spatial Audio, MagSafe charging case
Price Range: $199 - $249

Tone: Luxurious
Length: Medium (100 words)
Style: Formal
Include Call-to-Action: Yes

Description:
```

Optional fields (brand, category, audience, features, price) are only included if
filled in. The prompt always ends with `Description:` so it lines up with the marker
`evaluate_model.py` splits on, and with what the LoRA model was trained to continue.

## Copy-as-Python

"Copy as Python" doesn't just copy the raw prompt — it emits a runnable snippet wired to
this repo's own inference helpers:

```python
from src.inference.generate import load_generator, generate_text

generator = load_generator("lora_model")

prompt = """..."""

output = generator(
    prompt,
    max_length=150,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
    do_sample=True,
)
```

The sampling parameters come directly from the Advanced Generation Parameters sliders,
so what you preview in the UI is what the snippet will actually run.

## Running it locally

Any static file server works, e.g.:

```bash
cd prompt-builder
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Verification

Manually exercised in a browser: field input → live preview updates, every template
preset, chip selection, CTA toggle, all five sliders, the collapsible advanced panel,
copy-to-clipboard (prompt and Python), JSON download, save/load/delete/export/clear
history, theme toggle, and reset-all — no console errors in any of these paths.
