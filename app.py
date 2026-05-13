import os
import sys
import torch
import spaces
import gradio as gr
import gc
from huggingface_hub import snapshot_download

# ============================================================
# 预下载所有模型（CPU 操作，不触发 CUDA）
# ============================================================
BITDANCE_REPO = "BiliSakura/BitDance-14B-16x-diffusers"
print("📦 Pre-downloading BitDance repo...")
BITDANCE_LOCAL_PATH = snapshot_download(repo_id=BITDANCE_REPO)
print(f"📁 BitDance cached at: {BITDANCE_LOCAL_PATH}")

bitdance_pkg = os.path.join(BITDANCE_LOCAL_PATH, "bitdance_diffusers")
if os.path.isdir(bitdance_pkg):
    if BITDANCE_LOCAL_PATH not in sys.path:
        sys.path.insert(0, BITDANCE_LOCAL_PATH)
    print(f"✅ Added {BITDANCE_LOCAL_PATH} to sys.path")
else:
    print("⚠️ bitdance_diffusers directory not found")

ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
print("📦 Pre-downloading Z-Image-Turbo repo...")
ZIMAGE_LOCAL_PATH = snapshot_download(repo_id=ZIMAGE_REPO)
print(f"📁 Z-Image-Turbo cached at: {ZIMAGE_LOCAL_PATH}")

# ============================================================
# 模型注册表
# ============================================================
MODEL_REGISTRY = {
    "Z-Image-Turbo": {
        "default_steps": 9,
        "max_steps": 20,
        "default_guidance_scale": 0.0,
        "show_guidance": False,
        "description": "Ultra-fast AI image generation • 8 DiT forwards",
    },
    "BitDance-14B-16x": {
        "default_steps": 50,
        "max_steps": 100,
        "default_guidance_scale": 7.5,
        "show_guidance": True,
        "description": "High-quality 14B parameter model • Cinematic detail",
    },
}

MODEL_NAMES = list(MODEL_REGISTRY.keys())
DEFAULT_MODEL = "Z-Image-Turbo"

# ============================================================
# GPU Worker 内部的缓存（在 worker 进程中持久化）
# ============================================================
# 这些变量存在于 GPU worker 进程中，不在主进程
_gpu_cache = {
    "current_model": None,
    "pipe": None,
}


def _load_in_worker(model_name: str):
    """
    在 GPU worker 进程内部加载模型。
    利用 _gpu_cache 避免重复加载。
    """
    from diffusers import DiffusionPipeline

    if _gpu_cache["current_model"] == model_name and _gpu_cache["pipe"] is not None:
        print(f"⚡ [Worker] {model_name} already cached, reusing")
        return _gpu_cache["pipe"]

    # 卸载旧模型
    if _gpu_cache["pipe"] is not None:
        print(f"♻️ [Worker] Unloading {_gpu_cache['current_model']}...")
        del _gpu_cache["pipe"]
        _gpu_cache["pipe"] = None
        _gpu_cache["current_model"] = None
        gc.collect()
        torch.cuda.empty_cache()

    # 加载新模型
    if model_name == "Z-Image-Turbo":
        print("🔄 [Worker] Loading Z-Image-Turbo...")
        pipe = DiffusionPipeline.from_pretrained(
            ZIMAGE_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")

    elif model_name == "BitDance-14B-16x":
        print(f"🔄 [Worker] Loading BitDance-14B-16x...")
        pipe = DiffusionPipeline.from_pretrained(
            BITDANCE_LOCAL_PATH,
            custom_pipeline=BITDANCE_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    _gpu_cache["current_model"] = model_name
    _gpu_cache["pipe"] = pipe
    print(f"✅ [Worker] {model_name} loaded and cached!")
    return pipe


def _infer_in_worker(pipe, model_name, prompt, height, width, steps, cfg, seed):
    """在 GPU worker 进程内部执行推理"""
    generator = torch.Generator("cuda").manual_seed(seed)

    if model_name == "Z-Image-Turbo":
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        )
    elif model_name == "BitDance-14B-16x":
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            show_progress_bar=True,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return result.images[0]


# ============================================================
# 唯一的 @spaces.GPU 函数：加载 + 推理全在这里
# 返回 PIL Image（CPU 对象），不返回任何 CUDA tensor
# ============================================================
@spaces.GPU(duration=180)
def gpu_generate(
    model_name: str,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
):
    """
    完整的 GPU 流程：加载模型（如需）+ 推理。
    返回 PIL.Image（纯 CPU 对象，可安全传回主进程）。
    """
    # 步骤 1：在 worker 内加载/复用模型
    pipe = _load_in_worker(model_name)

    # 步骤 2：推理
    image = _infer_in_worker(
        pipe, model_name, prompt, height, width,
        num_inference_steps, guidance_scale, seed,
    )

    # 返回 PIL Image，这是纯 CPU 对象，不触发 CUDA 序列化
    return image


# ============================================================
# Gradio 入口函数（主进程，无 CUDA）
# ============================================================
def generate_image(
    model_name: str,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    randomize_seed: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate an image using the selected model.
    Args:
        model_name: Model to use. One of: "Z-Image-Turbo", "BitDance-14B-16x"
        prompt: Text description of the desired image.
        height: Image height in pixels (512–2048, step 64).
        width: Image width in pixels (512–2048, step 64).
        num_inference_steps: Number of denoising steps.
        guidance_scale: CFG scale. Use 0.0 for Z-Image-Turbo, 5–12 for BitDance.
        seed: Random seed. Ignored if randomize_seed is True.
        randomize_seed: If True, generate a random seed.
    Returns:
        Tuple of (generated_image, seed_used)
    """
    # ---- 输入验证（主进程，无 CUDA）----
    if not prompt or not prompt.strip():
        raise gr.Error("⚠️ Please enter a prompt!")

    if model_name not in MODEL_REGISTRY:
        raise gr.Error(f"❌ Unknown model: '{model_name}'. Available: {MODEL_NAMES}")

    config = MODEL_REGISTRY[model_name]
    height = int(max(512, min(2048, height)))
    width = int(max(512, min(2048, width)))
    num_inference_steps = int(max(1, min(config["max_steps"], num_inference_steps)))
    guidance_scale = float(max(0.0, min(20.0, guidance_scale)))

    if randomize_seed:
        import random
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed)
    prompt = prompt.strip()

    print(f"📋 Request | model={model_name} | steps={num_inference_steps} "
          f"| cfg={guidance_scale} | size={width}x{height} | seed={seed}")

    # ---- 调用 GPU 函数 ----
    image = gpu_generate(
        model_name=model_name,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    return image, seed


# ============================================================
# UI 动态更新
# ============================================================
def on_model_change(model_name):
    config = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[DEFAULT_MODEL])
    return (
        f"**{model_name}** — {config['description']}",
        gr.Slider(value=config["default_steps"], maximum=config["max_steps"]),
        gr.Slider(value=config["default_guidance_scale"], visible=config["show_guidance"]),
    )


# ============================================================
# 示例
# ============================================================
examples = [
    [
        "Z-Image-Turbo",
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, "
        "red floral forehead pattern. Elaborate high bun, golden phoenix headdress, "
        "red flowers, beads. Holds round folding fan with lady, trees, bird. "
        "Neon lightning-bolt lamp, bright yellow glow, above extended left palm. "
        "Soft-lit outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights.",
    ],
    [
        "Z-Image-Turbo",
        "A majestic dragon soaring through clouds at sunset, scales shimmering "
        "with iridescent colors, detailed fantasy art style",
    ],
    [
        "Z-Image-Turbo",
        "Cozy coffee shop interior, warm lighting, rain on windows, plants on shelves, "
        "vintage aesthetic, photorealistic",
    ],
    [
        "BitDance-14B-16x",
        "A close-up portrait in a cinematic photography style, capturing a girl-next-door "
        "look on a sunny daytime urban street. She wears a khaki sweater, with long, flowing "
        "hair gently draped over her shoulders. Her head is turned slightly, revealing soft "
        "facial features illuminated by realistic, delicate sunlight coming from the left. "
        "The sunlight subtly highlights individual strands of her hair. The image has a Canon "
        "film-like color tone, evoking a warm nostalgic atmosphere.",
    ],
    [
        "BitDance-14B-16x",
        "Astronaut riding a horse on Mars, cinematic lighting, sci-fi concept art, highly detailed",
    ],
    [
        "BitDance-14B-16x",
        "Portrait of a wise old wizard with a long white beard, holding a glowing crystal staff, "
        "magical forest background, oil painting style",
    ],
]

# ============================================================
# Theme
# ============================================================
custom_theme = gr.themes.Soft(
    primary_hue="yellow",
    secondary_hue="amber",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    text_size="lg",
    spacing_size="md",
    radius_size="lg",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    block_title_text_weight="600",
)

# ============================================================
# Build UI
# ============================================================
with gr.Blocks(fill_height=True, theme=custom_theme) as demo:
    gr.Markdown(
        """
        # 🎨 Multi-Model Image Generator
        **AI image generation** • Switch between models for speed or quality
        """,
        elem_classes="header-text",
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            model_selector = gr.Dropdown(
                choices=MODEL_NAMES,
                value=DEFAULT_MODEL,
                label="🤖 Model",
                info="Choose a generation model",
            )
            model_description = gr.Markdown(
                f"**{DEFAULT_MODEL}** — {MODEL_REGISTRY[DEFAULT_MODEL]['description']}"
            )

            prompt = gr.Textbox(
                label="✨ Your Prompt",
                placeholder="Describe the image you want to create...",
                lines=5, max_lines=10, autofocus=True,
            )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    height = gr.Slider(
                        minimum=512, maximum=2048, value=1024, step=64,
                        label="Height", info="Image height in pixels",
                    )
                    width = gr.Slider(
                        minimum=512, maximum=2048, value=1024, step=64,
                        label="Width", info="Image width in pixels",
                    )

                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=MODEL_REGISTRY[DEFAULT_MODEL]["max_steps"],
                    value=MODEL_REGISTRY[DEFAULT_MODEL]["default_steps"],
                    step=1, label="Inference Steps",
                    info="More steps → higher quality, slower",
                )

                guidance_scale = gr.Slider(
                    minimum=0.0, maximum=20.0,
                    value=MODEL_REGISTRY[DEFAULT_MODEL]["default_guidance_scale"],
                    step=0.5, label="Guidance Scale (CFG)",
                    info="How closely to follow the prompt (0 = off)",
                    visible=MODEL_REGISTRY[DEFAULT_MODEL]["show_guidance"],
                )

                with gr.Row():
                    randomize_seed = gr.Checkbox(label="🎲 Random Seed", value=True)
                    seed = gr.Number(label="Seed", value=42, precision=0)

                def toggle_seed(randomize):
                    return gr.Number(visible=not randomize)

                randomize_seed.change(toggle_seed, inputs=[randomize_seed], outputs=[seed])

            generate_btn = gr.Button(
                "🚀 Generate Image", variant="primary", size="lg", scale=1,
            )

            gr.Examples(
                examples=examples,
                inputs=[model_selector, prompt],
                label="💡 Try these prompts",
                examples_per_page=6,
            )

        with gr.Column(scale=1, min_width=320):
            output_image = gr.Image(
                label="Generated Image", type="pil", format="png",
                show_label=False, height=600,
                buttons=["download", "share"],
            )
            used_seed = gr.Number(
                label="🎲 Seed Used", interactive=False, container=True,
            )

    gr.Markdown(
        """
        ---
        <div style="text-align: center; opacity: 0.7; font-size: 0.9em; margin-top: 1rem;">
        <strong>Models:</strong>
        <a href="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo" target="_blank">Z-Image-Turbo</a> •
        <a href="https://huggingface.co/BiliSakura/BitDance-14B-16x-diffusers" target="_blank">BitDance-14B-16x</a>
        </div>
        """,
        elem_classes="footer-text",
    )

    model_selector.change(
        fn=on_model_change,
        inputs=[model_selector],
        outputs=[model_description, num_inference_steps, guidance_scale],
    )

    all_inputs = [
        model_selector, prompt, height, width,
        num_inference_steps, guidance_scale, seed, randomize_seed,
    ]
    all_outputs = [output_image, used_seed]

    generate_btn.click(fn=generate_image, inputs=all_inputs, outputs=all_outputs)
    prompt.submit(fn=generate_image, inputs=all_inputs, outputs=all_outputs)


if __name__ == "__main__":
    demo.launch(
        css="""
        .header-text h1 {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header-text p {
            font-size: 1.1rem !important;
            color: #64748b !important;
            margin-top: 0 !important;
        }
        .footer-text { padding: 1rem 0; }
        .footer-text a {
            color: #f59e0b !important;
            text-decoration: none !important;
            font-weight: 500;
        }
        .footer-text a:hover { text-decoration: underline !important; }
        @media (max-width: 768px) {
            .header-text h1 { font-size: 1.8rem !important; }
            .header-text p { font-size: 1rem !important; }
        }
        button, .gr-button { transition: all 0.2s ease !important; }
        button:hover, .gr-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        }
        .gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
        """,
        footer_links=["api", "gradio"],
        mcp_server=True,
    )
