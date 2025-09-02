# pip install -U google-genai pillow

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List, Dict, Optional
from pathlib import Path
from io import BytesIO
import threading
import argparse
import textwrap
import random
import shutil
import base64
import json
import time
import os

from google.genai import errors
from google import genai
from PIL import Image

SafetyList = List[Dict[str, str]]

class GeminiBatchImageGenerator:
    def __init__(self,  api_key_file, input_folder, output_folder, model, top_p, top_k, temperature, seed, safety_mode, save_text,
                 save_origin, api_rate_limit, max_attempts_count, prompt=None, safety_settings = None, **kwargs):

        # Initialize instance variables
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.seed = seed
        self.safety_mode = safety_mode
        self.safety_settings = safety_settings
        self.save_text = save_text
        self.save_origin = save_origin
        self.api_rate_limit = api_rate_limit
        self.max_attempts_count = max_attempts_count
        self.model = model

        # Load API key and default prompt
        self.api_key = self._load_api_key(api_key_file)
        self.client = genai.Client(api_key=self.api_key)

        if prompt is None:
            self.default_prompt = self._load_default_prompt("default_prompt.txt")
        else:
            self.default_prompt = prompt

        self.gen_config = self._make_gen_config()




    def set_attr(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: {k} is not a valid attribute.")
        self.gen_config = self._make_gen_config()
        return self._make_gen_config()


    def _load_api_key(self, file_path: str) -> str:
        p = Path(file_path)
        try:
            return p.read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError:
            return p.read_text(encoding="utf-16").strip()

    def _load_default_prompt(self, path: str) -> str:
        # UTF-8 및 UTF-8 BOM 자동 처리(utf-8-sig). 줄 단위로 읽어 공백/빈줄 제거 후 공백 하나로 합치기.
        text = Path(path).read_text(encoding="utf-8-sig")
        merged = " ".join(line.strip() for line in text.splitlines() if line.strip())
        return textwrap.dedent(merged).strip()

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

    def _base_out(self, input_img: Union[str, Path, Image.Image], suffix: str = "__gemini", default_ext: str = ".png") -> tuple[Path, str, str]:
        """
        Build base output directory and name components.

        Returns:
            (out_dir, base_stem, ext)
            - out_dir: Path to output directory
            - base_stem: base filename stem with suffix applied
            - ext: extension to use for images
        """
        out_dir = Path(self.output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(input_img, (str, Path)):
            in_path = Path(input_img)
            ext = in_path.suffix if in_path.suffix else default_ext
            stem = in_path.stem + suffix
        else:
            ext = default_ext
            stem = "image" + suffix
        return out_dir, stem, ext

    def _build_safety(self) -> Optional[SafetyList]:
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
        if self.safety_settings:
            return self.safety_settings

        if not self.safety_mode:
            return None

        if self.safety_mode.lower() == "off":
            thr = "BLOCK_NONE"
        elif self.safety_mode.lower() == "relaxed":
            thr = "BLOCK_ONLY_HIGH"
        elif self.safety_mode.lower() == "balanced":
            thr = "BLOCK_MEDIUM_AND_ABOVE"
        elif self.safety_mode.lower() == "strict":
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

    def _make_gen_config(self, candidate_count: Optional[int] = None, max_output_tokens: Optional[int] = None):
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
            resp_modalities = ["IMAGE", "TEXT"] if self.save_text else ["IMAGE"]

        cfg: Dict[str, object] = {"response_modalities": resp_modalities}

        # Sampling / decoding params (only include if provided)
        if self.temperature is not None:
            cfg["temperature"] = float(self.temperature)
        if self.top_p is not None:
            cfg["top_p"] = float(self.top_p)
        if self.top_k is not None:
            cfg["top_k"] = int(self.top_k)
        if self.seed is not None:
            cfg["seed"] = int(self.seed)
        if candidate_count is not None:
            cfg["candidate_count"] = int(candidate_count)
        if max_output_tokens is not None:
            cfg["max_output_tokens"] = int(max_output_tokens)

        # Optional safety settings (mode or explicit list)
        s = self._build_safety()
        if s:
            cfg["safety_settings"] = s
        return cfg
    
    def _make_gen_config_with_seed(self, seed=None, candidate_count=None, max_output_tokens=None):
        cfg = self._make_gen_config(candidate_count, max_output_tokens)
        cfg["seed"] = int(seed if seed is not None else random.randint(1, 2**31 - 1))
        return cfg

    def get_image(self, input_img):
        pil_img = self._ensure_pil(input_img)
        prev_size = pil_img.size

        if pil_img.width > 3072 or pil_img.height > 3072:
            scale = min(3072 / pil_img.width, 3072 / pil_img.height)

            pil_img = pil_img.resize((int(pil_img.size[0] * scale), int(pil_img.size[1] * scale)), Image.LANCZOS)
        # if pil_img.size != prev_size:
        #     print(f"Down sized image : {prev_size}  --> {pil_img.size}")

        return pil_img

    def save_all_images(self, resp, out_dir, base_stem, ext, seed_idx):
        saved_path, text_chunk = [], []
        img_idx = 0
        txt_path = out_dir / f"{base_stem}.txt"
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
                    saved_path.append(out_path)
                    txt_path = out_dir / f"{base_stem}_{img_idx}.txt"
                # Text parts
                if getattr(p, "text", None):
                    text_chunk.append(p.text)

        # Log results
        txt_saved = None
        if saved_path:
            print(f"[OK] Saved {len(saved_path)} image(s) in {seed_idx+1} attempt(s):")
            for pth in saved_path:
                print("  -", pth.resolve())
            if self.save_text and text_chunk:
                txt_path.write_text("\n\n---\n\n".join(text_chunk), encoding="utf-8")
                txt_saved = txt_path
            

        return saved_path, txt_saved
        

    # ---------- main ----------
    def make_image(self, input_img: Union[str, Path, Image.Image, None], prompt: Optional[str] = None):
        if prompt is None:
            prompt = self.default_prompt

        # Determine base output naming
        out_dir, base_stem, ext = self._base_out(input_img)

        # Build input contents: text + optional image
        contents = [prompt]
        image = self.get_image(input_img)
        contents.append(image)

        try:
            # Call the model
            for seed_idx in range(self.max_attempts_count):
                # Compose config (modalities, safety, and sampling)
                gen_config = self.gen_config
                if seed_idx > 0:
                    gen_config = self._make_gen_config_with_seed()

                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=gen_config,
                )

                blocked = getattr(resp, "prompt_feedback", None) and getattr(resp.prompt_feedback, "block_reason", None)
                save_path, text_chunks = self.save_all_images(resp, out_dir, base_stem, ext, seed_idx)
                if save_path:  #  Success to save images
                    return save_path, text_chunks

                # print(f"[WARN] Request blocked: {blocked}}")
                time.sleep(1 + random.random() * 2)  # Waiting before retrying

            return None, None

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


    def processing(self, input_img, api_semaphore):
                   
        """병렬 처리를 위한 래퍼 함수 - API 제한 적용"""
        with api_semaphore:  # 동시 API 호출 수 제한
            time.sleep(self.api_rate_limit)  # API 호출 간격 제한
            
            saved_paths, _ = self.make_image(
                input_img=input_img,
            )
            
            not_generated_files = []
            generated_files = saved_paths
            if not saved_paths:
                not_generated_files.append(input_img)
                os.makedirs(f"fail/{self.input_folder}", exist_ok=True)
                shutil.copy(input_img, f"fail/{self.input_folder}/{os.path.basename(input_img)}")
                print("[FAIL] Failed to generate image")
                print(f"  - {input_img}")
            elif self.save_origin:
                shutil.copy(input_img, f"{self.output_folder}/{os.path.basename(input_img)}")
                    
            return not_generated_files, generated_files, input_img  # 파일명도 함께 반환
                


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Gemini Flash Image Generation/Editing Example")
    parser.add_argument("--output-folder", default="Output", help="Directory to save output images")
    parser.add_argument("--input-folder", default="Inputs", help="Path to the input image file")
    parser.add_argument("--api-key-file", default="./gemini_api_key.txt", help="Path to the API key file")
    parser.add_argument("--model", default="gemini-2.5-flash-image-preview", help="Model to use for image generation")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability")
    parser.add_argument("--top-k", type=float, default=None, help="Nucleus sampling probability")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--safety-mode", default="off", choices=["off", "relaxed", "balanced", "strict"], help="Safety mode")
    parser.add_argument("--save-text", action="store_true", help="Whether to save text captions")
    parser.add_argument("--save-args", action="store_true", help="Whether to save options")
    parser.add_argument("--save-origin", action="store_true", help="Whether to save original image")
    parser.add_argument("--prompt", type=str, default=None, help="Override default prompt")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum number of concurrent threads")
    parser.add_argument("--api-rate-limit", type=float, default=0.5, help="Minimum interval between API calls (seconds)")
    parser.add_argument("--max-attempts-count", type=int, default=1, help="Maximum number of attempts")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(1, 2**31 - 1)


    g = GeminiBatchImageGenerator(**vars(args))

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
    

    os.makedirs(args.output_folder, exist_ok=True)
    if args.save_args:
        if args.prompt == None:
            args.prompt = g.default_prompt
        # Save the options used for this file
        json_path = Path(args.output_folder) / f"options.json"
        idx = 1
        while os.path.exists(json_path):
            json_path = Path(args.output_folder) / f"options_{idx}.json"
            idx += 1
        now = time.localtime()
        args.create_time = time.strftime("%Y-%m-%d %H:%M:%S", now)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(vars(args), jf, ensure_ascii=False, indent=4)

    # API 호출 제한을 위한 세마포어 생성
    api_semaphore = threading.Semaphore(args.max_workers)
    not_generated_files = []
    generated_files = []
    # 병렬 처리 with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 모든 작업 제출
        future_to_file = {
            executor.submit(
                g.processing,
                input_img=file,
                api_semaphore=api_semaphore,
            ): file for file in files
        }
        
        # 완료된 작업들 처리
        completed = 0
        check_file = []
        for future in as_completed(future_to_file):
            check_file.append(future)
            file = future_to_file[future]
            completed += 1
            try:
                failed_files, gen_files, processed_file = future.result()
                not_generated_files.extend(failed_files)
                generated_files.extend(gen_files)
            except Exception as e:
                not_generated_files.append(file)
                # print(f"[{completed}/{len(files)}] Failed: {os.path.basename(file)} - {e}")

    if args.save_args:
        args.generated_files = [str(d) for d in generated_files]
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(vars(args), jf, ensure_ascii=False, indent=4)


    print(f"\nProcessing completed!")
    print(f"Successfully processed: {len(files) - len(set(not_generated_files))}/{len(files)}")
    if not_generated_files:
        print(f"Failed files: ")
        for f in set(not_generated_files):
            print(f"{os.path.join(args.input_folder, os.path.basename(f))}")
    else:
        print("All files processed successfully!")