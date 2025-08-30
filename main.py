# pip install -U google-genai pillow

from typing import Union, List, Dict, Optional
from google.genai import errors
from google import genai
from pathlib import Path
from io import BytesIO
from PIL import Image
import argparse
import textwrap
import random
import shutil
import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SafetyList = List[Dict[str, str]]

class Gemini:
    def __init__(self, api_key_file: Union[str, Path], model: str = "gemini-2.5-flash-image-preview"):
        # Read API key from file
        key_path = Path(api_key_file)
        # Handle potential UTF-16 BOM encoding
        try:
            self.api_key = key_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            self.api_key = key_path.read_text(encoding="utf-16").strip()

        # Select model (2.5 image preview by default)
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

        # Default prompt (feel free to override on call)
        self.default_prompt = open("default_prompt.txt", "rb").readlines()
        self.default_prompt = " ".join([
                                            line.decode("utf-8").strip()  # decode 추가
                                            for line in self.default_prompt
                                            if line.strip()
                                        ])

        self.default_prompt = textwrap.dedent(self.default_prompt).strip()

    def _ensure_bytes(self, data):
        # Google SDK가 str(base64) 또는 bytes를 줄 수 있으니 모두 처리
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, str):
            try:
                return base64.b64decode(data, validate=True)
            except Exception:
                # 혹시 base64가 아니면 그대로 시도하게 냅둔다
                return data.encode("utf-8", errors="ignore")
        return bytes(data)

    def _looks_like_image(self, b):
        if not b or len(b) < 8:
            return False
        sig = b[:12]
        # PNG, JPEG, WEBP, GIF, BMP 등 간단 시그니처 체크
        return (
            sig.startswith(b"\x89PNG\r\n\x1a\n") or
            sig.startswith(b"\xFF\xD8\xFF") or
            (sig[:4] == b"RIFF" and sig[8:12] == b"WEBP") or
            sig.startswith(b"GIF8") or
            sig.startswith(b"BM")
        )

    # ---------- utils ----------
    def _ensure_pil(self, img_or_path: Union[str, Path, Image.Image]) -> Image.Image:
        """Return a PIL.Image regardless of whether input is a path or already an Image."""
        if isinstance(img_or_path, Image.Image):
            return img_or_path
        return Image.open(str(img_or_path))

    def _base_out(
        self,
        input_img: Union[str, Path, Image.Image],
        output_folder: Union[str, Path] = "Output",
        suffix: str = "_gemini",
        default_ext: str = ".png"
    ) -> tuple[Path, str, str]:
        """
        Build base output directory and name components.

        Returns:
            (out_dir, base_stem, ext)
            - out_dir: Path to output directory
            - base_stem: base filename stem with suffix applied
            - ext: extension to use for images
        """
        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(input_img, (str, Path)):
            in_path = Path(input_img)
            ext = in_path.suffix if in_path.suffix else default_ext
            stem = in_path.stem + suffix
        else:
            ext = default_ext
            stem = "image" + suffix
        return out_dir, stem, ext

    def _build_safety(self, mode: Optional[str] = None, explicit: Optional[SafetyList] = None) -> Optional[SafetyList]:
        """
        Build safety settings.

        Args:
            mode: one of {"off","relaxed","balanced","strict"}.
                  Mapped thresholds:
                    off      -> BLOCK_NONE
                    relaxed  -> BLOCK_ONLY_HIGH
                    balanced -> BLOCK_MEDIUM_AND_ABOVE
                    strict   -> BLOCK_LOW_AND_ABOVE
            explicit: if provided, a list of dicts like
                      {"category": "...", "threshold": "..."} will be passed as-is.

        Returns:
            A list of safety settings or None.
        """
        if explicit:
            return explicit

        if not mode:
            return None

        if mode.lower() == "off":
            thr = "BLOCK_NONE"
        elif mode.lower() == "relaxed":
            thr = "BLOCK_ONLY_HIGH"
        elif mode.lower() == "balanced":
            thr = "BLOCK_MEDIUM_AND_ABOVE"
        elif mode.lower() == "strict":
            thr = "BLOCK_LOW_AND_ABOVE"
        else:
            raise ValueError("Unknown safety mode. Choose: off, relaxed, balanced, strict")

        cats = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
        ]
        return [{"category": c, "threshold": thr} for c in cats]

    def _make_gen_config(
        self,
        want_text: bool,
        safety_mode: Optional[str],
        safety_list: Optional[SafetyList],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        candidate_count: Optional[int] = None,
        max_output_tokens: Optional[int] = None
    ):
        """
        Compose generation config including modalities, safety, and sampling parameters.
        Notes:
          - Some models (e.g., models/gemini-2.0-flash-preview-image-generation) do NOT support multiple candidates.
            If you pass candidate_count > 1 there, the API may return INVALID_ARGUMENT.
        """
        # Response modalities by model
        if self.model.startswith("models/gemini-2.0"):
            # Gemini 2.0 preview image generation tends to be stable with BOTH IMAGE and TEXT
            resp_modalities = ["IMAGE", "TEXT"]
        else:
            resp_modalities = ["IMAGE", "TEXT"] if want_text else ["IMAGE"]

        cfg: Dict[str, object] = {"response_modalities": resp_modalities}

        # Sampling / decoding params (only include if provided)
        if temperature is not None:
            cfg["temperature"] = float(temperature)
        if top_p is not None:
            cfg["top_p"] = float(top_p)
        if top_k is not None:
            cfg["top_k"] = int(top_k)
        if seed is not None:
            cfg["seed"] = int(seed)
        if candidate_count is not None:
            cfg["candidate_count"] = int(candidate_count)
        if max_output_tokens is not None:
            cfg["max_output_tokens"] = int(max_output_tokens)

        # Optional safety settings (mode or explicit list)
        s = self._build_safety(safety_mode, safety_list)
        if s:
            cfg["safety_settings"] = s
        return cfg

    # ---------- main ----------
    def make_image(
        self,
        input_img: Union[str, Path, Image.Image, None],
        prompt: Optional[str] = None,
        output_folder: Union[str, Path] = "Output",
        save_text: bool = True,
        safety_mode: Optional[str] = "off",            # "off" | "relaxed" | "balanced" | "strict"
        safety_settings: Optional[SafetyList] = None, # overrides safety_mode if provided
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        candidate_count: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Generate or edit an image with the selected model.

        Behavior:
          - Saves ALL image parts found in the response:
              output/<stem>_gemini_1<ext>, output/<stem>_gemini_2<ext>, ...
          - Optionally saves text captions to:
              output/<stem>_gemini.txt
          - Safety can be controlled via `safety_mode` or `safety_settings`.
          - Sampling can be controlled with temperature/top_p/top_k/seed/candidate_count/max_output_tokens.
        """
        if prompt is None:
            prompt = self.default_prompt

        # Determine base output naming
        out_dir, base_stem, ext = self._base_out(input_img, output_folder=output_folder)

        # Build input contents: text + optional image
        contents = [prompt]
        if input_img is not None:
            pil_img = self._ensure_pil(input_img)
            prev_size = pil_img.size

            if pil_img.width > 1536 or pil_img.height > 1536:
                scale = min(1536 / pil_img.width, 1536 / pil_img.height)

                pil_img = pil_img.resize((int(pil_img.size[0] * scale), int(pil_img.size[1] * scale)), Image.LANCZOS)
            if pil_img.size != prev_size:
                print(f"Down sized image : {prev_size}  --> {pil_img.size}")

            contents.append(pil_img)

        # Compose config (modalities, safety, and sampling)
        gen_config = self._make_gen_config(
            want_text=save_text,
            safety_mode=safety_mode,
            safety_list=safety_settings,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            candidate_count=candidate_count,
            max_output_tokens=max_output_tokens,
        )

        try:
            # Call the model
            resp = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=gen_config,
            )

            saved_paths = []
            text_chunks = []

            # Save all images and optionally the text parts
            img_idx = 0
            txt_path = out_dir / f"{base_stem}.txt"
            json_path = out_dir / f"{base_stem}.json"
            for cand in getattr(resp, "candidates", []):
                parts = getattr(cand, "content", None)
                parts = getattr(parts, "parts", []) if parts else []
                for p in parts:
                    # Image parts
                    if getattr(p, "inline_data", None) and getattr(p.inline_data, "mime_type", "").startswith("image/"):
                        raw = self._ensure_bytes(p.inline_data.data)

                        if not self._looks_like_image(raw):
                            dbg_path = out_dir / f"{base_stem}_invalid_{img_idx+1}.bin"
                            with open(dbg_path, "wb") as f:
                                f.write(raw if isinstance(raw, (bytes, bytearray)) else bytes(raw))
                            print(f"[WARN] Not a valid image signature. Dumped bytes to: {dbg_path}")
                            continue


                        img_idx += 1
                        img = Image.open(BytesIO(raw))
                        img.load()
                        out_path = out_dir / f"{base_stem}_{img_idx}{ext}"
                        while os.path.exists(out_path):
                            img_idx += 1
                            out_path = out_dir / f"{base_stem}_{img_idx}{ext}"
                        img.save(out_path)
                        saved_paths.append(out_path)
                        txt_path = out_dir / f"{base_stem}_{img_idx}.txt"
                        json_path = out_dir / f"{base_stem}_{img_idx}.json"
                    # Text parts
                    if getattr(p, "text", None):
                        text_chunks.append(p.text)

            # Save caption file if requested
            txt_saved = None
            if save_text and text_chunks:
                txt_path.write_text("\n\n---\n\n".join(text_chunks), encoding="utf-8")
                txt_saved = txt_path

            # Log results
            if saved_paths:
                print(f"[OK] Saved {len(saved_paths)} image(s):")
                for pth in saved_paths:
                    print("  -", pth.resolve())
                
            else:
                # print("[INFO] No image parts returned.")
                return None, None


            return saved_paths, txt_saved

        except errors.APIError as e:
            # Basic error diagnostics (helpful for quota/safety issues)
            print(f"[APIError] code={getattr(e,'code',None)} message={getattr(e,'message','')}")
            print(f"\tmessage: {getattr(e, 'message', '')}")
            print(f"\tdetails: {getattr(e, 'details', '')}")
            data = getattr(e, "response_json", {}) or {}
            details = data.get("error", {}).get("details", [])
            for d in details:
                t = d.get("@type","")
                if t.endswith("QuotaFailure"):
                    viols = [v.get("quotaId","") for v in d.get("violations",[])]
                    print("[Quota Violations]", viols)
                if t.endswith("RetryInfo") and "retryDelay" in d:
                    print("[RetryInfo]", d["retryDelay"])
            return None, None


    def processing(self, input_img, 
                   input_folder,
                   output_folder,
                   save_origin,
                   save_option,
                   api_semaphore,
                   api_rate_limit,
                   **kwargs):
        """병렬 처리를 위한 래퍼 함수 - API 제한 적용"""
        with api_semaphore:  # 동시 API 호출 수 제한
            try:
                time.sleep(api_rate_limit)  # API 호출 간격 제한
                
                saved_paths, _ = self.make_image(
                    input_img=input_img,
                    output_folder=output_folder,
                    **kwargs
                )
                
                not_generated_files = []
                if not saved_paths:
                    not_generated_files.append(input_img)
                    os.makedirs(f"fail/{input_folder}", exist_ok=True)
                    shutil.copy(input_img, f"fail/{input_folder}/{os.path.basename(input_img)}")
                    print("[FAIL] No Image")
                    print(f"  - {input_img}")
                elif save_origin:
                    shutil.copy(input_img, f"{output_folder}/{os.path.basename(input_img)}")
                elif save_option:
                    # Save the options used for this file
                    json_path = saved_paths[0].with_suffix(".json")
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(kwargs, jf, ensure_ascii=False, indent=4)
                        
                return not_generated_files, input_img  # 파일명도 함께 반환
                
            except Exception as e:
                print(f"[ERROR] Failed to process {input_img}: {e}")
                return [input_img], input_img
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Gemini Flash Image Generation/Editing Example")
    parser.add_argument("--output-folder", default="Output", help="Directory to save output images")
    parser.add_argument("--input-folder", default="Inputs", help="Path to the input image file")
    parser.add_argument("--api-key-file", default="./gemini_api_key.txt", help="Path to the API key file")
    parser.add_argument("--model", default="gemini-2.5-flash-image-preview", help="Model to use for image generation")
    parser.add_argument("--top-p", type=float, default=0.7, help="Nucleus sampling probability")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--safety-mode", default="off", choices=["off", "relaxed", "balanced", "strict"], help="Safety mode")
    parser.add_argument("--save-text", action="store_true", help="Whether to save text captions")
    parser.add_argument("--save-option", action="store_true", help="Whether to save options")
    parser.add_argument("--save-origin", action="store_true", help="Whether to save original image")
    parser.add_argument("--prompt", type=str, default=None, help="Override default prompt")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum number of concurrent threads")
    parser.add_argument("--api-rate-limit", type=float, default=0.5, help="Minimum interval between API calls (seconds)")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(1, 2**31 - 1)


    g = Gemini(api_key_file=args.api_key_file, model=args.model)

    EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif", ".avif"}
    files = sorted(
            str(p) for p in Path(args.input_folder).rglob("*")
            if p.is_file() and p.suffix.lower() in EXTS
    )
    if not files:
        print(f"No image files found in {args.input_folder}")
        exit(0)
    
    print(f"Found {len(files)} image files. Processing with max {args.max_workers} concurrent threads...")
    print(f"API rate limit: {args.api_rate_limit}s between calls")
    
    # API 호출 제한을 위한 세마포어 생성
    api_semaphore = threading.Semaphore(args.max_workers)
    
    not_generated_files = []
    for _ in range(1):
        # 병렬 처리 with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # 모든 작업 제출
            future_to_file = {
                executor.submit(
                    g.processing,
                    input_img=file,
                    api_semaphore=api_semaphore,
                    **vars(args),
                ): file for file in files
            }
            
            # 완료된 작업들 처리
            completed = 0
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                completed += 1
                try:
                    failed_files, processed_file = future.result()
                    not_generated_files.extend(failed_files)
                except Exception as e:
                    not_generated_files.append(file)
                    print(f"[{completed}/{len(files)}] Failed: {os.path.basename(file)} - {e}")

        print(f"\nProcessing completed!")
        print(f"Successfully processed: {len(files) - len(not_generated_files)}/{len(files)}")
        if not_generated_files:
            print(f"Failed files: {set(os.path.basename(f) for f in not_generated_files)}")
        else:
            print("All files processed successfully!")