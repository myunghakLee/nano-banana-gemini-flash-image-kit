# Gemini-flash-image-kit(old: nano banana)

![alt text](figure/figure1.png)

Generate or edit images with **Gemini 2.5 Flash Image Preview** using a simple Python script that:

- reads your API key from a file,
- scans an input folder (recursive),
- calls the API with adjustable **safety** and **sampling** options,
- saves **every returned image**, optional **captions**, and (optionally) the **CLI options** used.

> Entry point: `python main.py`

## Features

- **Folder batch**: processes every image under `Inputs/` (recursive) by default
- **Save all images**: not just the first — results go to `Outputs/`
- **Predictable names**: `<stem>_gemini_{index}<ext>` (e.g., `photo_gemini_1.png`)
- **Captions (optional)**: `--save-text` writes a `.txt` per input (suffix matches the last saved image index)
- **Save options (optional)**: `--save-option` writes a `.json` of CLI arguments next to the _first_ saved image
- **Safety presets**: `off | relaxed | balanced | strict` (mapped to API thresholds)
- **Sampling controls**: `--top-p`, `--temperature`, `--seed` (random if omitted)
- **Wide input support**: `.png .jpg .jpeg .webp .bmp .gif .tif .tiff .heic .heif .avif`

---

## Requirements

- Python **3.9+**
- Packages: `google-genai`, `pillow`

```bash
pip install -U google-genai pillow
```

---

## Setup

1. Create an API key in Google AI Studio / Cloud Console.
2. Save the raw key string into **`gemini_api_key.txt`** at the project root (or pass a custom path via `--api-key-file`).

> Keep keys out of source control; add `gemini_api_key.txt` to `.gitignore`.

---

## Project Layout

```
project/
├─ main.py                 # this script
├─ gemini_api_key.txt      # your API key (do NOT commit)
├─ Inputs/                 # put source images here
└─ Outputs/                # results are written here
```

---

## Quick Start

1. Get an API key from Google AI Studio / Cloud Console: [Link](https://console.cloud.google.com/)
2. Write down your API key into `gemini_api_key.txt` file.
3. Put one or more images under `Inputs/`.
4. Run the script:

```bash
python main.py
```

5. Check `Outputs/` for generated images (and optional `.txt` / `.json`).

**Default model**: `gemini-2.5-flash-image-preview`

---

## CLI Usage

```bash
python main.py \
  --outputs-dir Outputs \
  --input-file Inputs \
  --api-key-file ./gemini_api_key.txt \
  --model gemini-2.5-flash-image-preview \
  --top-p 0.7 \
  --temperature 0.6 \
  --seed 1234567 \
  --safety-mode balanced \
  --save-text \
  --save-option \
  --prompt "Natural daylight, full‑body figure on a computer desk; faces crisp; subtle reflections"
```

**Arguments**

- `--outputs-dir` (str, default: `Outputs`) — destination directory
- `--input-file` (str/path, default: `Inputs`) — process a **file** or **directory** (recursive)
- `--api-key-file` (str/path, default: `./gemini_api_key.txt`) — file that contains your API key
- `--model` (str, default: `gemini-2.5-flash-image-preview`) — model name
- `--top-p` (float, default: `0.7`) — nucleus sampling
- `--temperature` (float, default: `0.6`) — sampling temperature (↑ = more diversity)
- `--seed` (int, default: random) — if omitted, a random seed is generated at runtime
- `--safety-mode` (`off|relaxed|balanced|strict`, default: `off`) — safety preset
- `--save-text` (flag) — also write a caption file for text parts returned by the model
- `--save-option` (flag) — save a JSON dump of the CLI args next to the _first_ saved image for each input
- `--prompt` (str, default: built‑in) — override the script’s default prompt

> Tip: run `python main.py -h` to see help.

---

## Safety Modes

The `--safety-mode` flag maps to the following thresholds across categories:

- `off` → `BLOCK_NONE`
- `relaxed` → `BLOCK_ONLY_HIGH`
- `balanced` → `BLOCK_MEDIUM_AND_ABOVE`
- `strict` → `BLOCK_LOW_AND_ABOVE`

Applied to:

- `HARM_CATEGORY_HARASSMENT`
- `HARM_CATEGORY_HATE_SPEECH`
- `HARM_CATEGORY_SEXUALLY_EXPLICIT`
- `HARM_CATEGORY_DANGEROUS_CONTENT`
- `HARM_CATEGORY_CIVIC_INTEGRITY`

---

## Default Prompt

If you don’t provide `--prompt`, this prompt is used:

```text
Ultra realistic collectible figure/statue of [character / style / pose] naturally placed on a computer desk,
entire figure fully in frame including the base (no cropping).
Primary focus on the face: eyes crisp with natural catchlights, clean paint masks around eyes and mouth,
facial features clearly readable, no smoothing or waxy look.
```

---
