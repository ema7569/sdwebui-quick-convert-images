import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from modules import scripts

import gradio as gr
from modules.images import read_info_from_image, save_image_with_geninfo
from modules.paths_internal import default_output_dir
from modules.script_callbacks import on_ui_tabs
from PIL import Image

EXT_DIR = scripts.basedir()
FOLDER_FILE = os.path.join(EXT_DIR, "folders_user.txt")

def default_folders_value():
    base_dir = os.getcwd()
    default_folders = ["outputs", "models\\Lora", "models\\Stable-diffusion", "embeddings"]
    return "\n".join([os.path.join(base_dir, f) for f in default_folders])

def get_folder_list_from_ui():
    if not os.path.exists(FOLDER_FILE):
        return default_folders_value()
    with open(FOLDER_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    return content  # retourne exactement ce qu'il y a dans le fichier, même vide

def save_folders(text):
    with open(FOLDER_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    return f"[INFO] Saved {len(text.splitlines())} folders."


def __process(paths: list, ext_from: list, ext_to: str, cpu: int,
              recursive: bool, delete: bool, force: bool,
              size: str = None, resize: bool = True, crop: bool = True):

    from PIL import ImageOps

    #print(f"[DEBUG] recursive={recursive} ({type(recursive)})")

    # --- Parsing de la taille cible ---
    target_size = None
    if size:
        try:
            w, h = [int(x) for x in size.lower().replace(" ", "").split("x")]
            target_size = (w, h)
        except Exception:
            print(f"[WARN] Invalid size format: '{size}'. Expected 'WxH'.")

    all_files = []
    for path in paths:
        path = path.strip('"').strip()
        if os.path.isfile(path):
            if any(path.endswith(ext) for ext in ext_from):
                all_files.append(os.path.normpath(path))
        elif os.path.isdir(path):
            for ext in ext_from:
                files = glob(os.path.join(path, "**", f"*.{ext}"), recursive=recursive)
                all_files.extend([f for f in files if f.lower().endswith(ext.lower())])
        else:
            print(f'[WARN] Path "{path}" does not exist')
            continue

    if not all_files:
        print("[INFO] All files processed")
        return

    print(f"[INFO] Processing {len(all_files)} files, please hold...")


    def _process(file_path: str, target_size=None, resize=True, crop=True):
        from PIL import ImageOps

        try:
            with Image.open(file_path) as img:
                # --- Read info ---
                info, _ = read_info_from_image(img)
                if info is None or not info:
                    if not force:
                        return

                # --- Resize / Crop ---
                if target_size:
                    original_size = img.size
                    if resize and crop:
                        img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
                    elif resize:
                        ratio_w = target_size[0] / img.width
                        ratio_h = target_size[1] / img.height
                        ratio = min(ratio_w, ratio_h)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    elif crop:
                        left = max((img.width - target_size[0]) // 2, 0)
                        top = max((img.height - target_size[1]) // 2, 0)
                        right = left + target_size[0]
                        bottom = top + target_size[1]
                        img = img.crop((left, top, right, bottom))
                    print(f"[OK] {file_path} resized from {original_size} to {img.size}")

                # --- Conversion ---
                ext_matched = next((ext for ext in ext_from if file_path.endswith(ext)), None)
                if ext_matched:
                    target_path = file_path.replace(ext_matched, ext_to)

                    # Assurer un mode compatible pour JPG/PNG
                    if img.mode not in ("RGB", "RGBA", "L"):
                        img = img.convert("RGB")

                    save_image_with_geninfo(img, info, target_path)

        except Image.DecompressionBombError:
            print(f'[SKIP] "{file_path}" DecompressionBombError')
            return
        except Exception as e:
            print(f'[ERROR] Cannot process "{file_path}": {e}')
            return

        # --- Suppression du fichier source après fermeture ---
        target_path = os.path.splitext(file_path)[0] + f".{ext_to.lower()}"
        if delete and os.path.normpath(file_path) != os.path.normpath(target_path):
            try:
                os.remove(file_path)
                print(f"[DEL] Deleted {file_path}")
            except Exception as e:
                print(f"[WARN] Failed to delete {file_path}: {e}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with ThreadPoolExecutor(max_workers=max(1, int(cpu))) as executor:
            for file in all_files:
                # --- IMPORTANT : passer size/resize/crop à _process ---
                executor.submit(_process, file, target_size, resize, crop)

    print("[DONE] All files processed successfully.")


# --- UI Gradio ---
def QuickConvertImg():
    with gr.Blocks() as REEE:
        group_a = gr.Group(elem_classes="rewrite-group")
        group_a.__enter__()

        with gr.Column():
            save_output = gr.Textbox(label="Log", interactive=False)
            folders_input = gr.TextArea(
                value=get_folder_list_from_ui(),
                label="Folders (one per line)",
                placeholder="Paste folder paths here...",
                lines=5
            )
            folders_input.do_not_save_to_config = True

            with gr.Row():
                gr.Markdown("", scale=1)
                save_btn = gr.Button("Save Folders List")
                save_btn.click(fn=save_folders, inputs=[folders_input], outputs=[save_output])

                reload_btn = gr.Button("Reload Folders")
                reload_btn.click(
                    fn=lambda: get_folder_list_from_ui(),  # lit le fichier à chaque click
                    inputs=[],
                    outputs=[folders_input]                # met à jour le TextArea
                )

            # --- Extensions source / cible ---
            with gr.Row():
                ext_from = gr.Dropdown(
                    label="Source Extensions",
                    info="from",
                    choices=["png", "jpg", "jpeg", "webp", "avif"],
                    value=["png", "jpeg", "webp"],
                    multiselect=True
                )
                ext_to = gr.Dropdown(
                    value="jpg",
                    label="Target Extension",
                    info="to",
                    choices=["png", "jpg", "jpeg", "webp", "avif"]
                )
            with gr.Row():
                cpu = gr.Number(value=4, step=1, label="Concurrency", info="processes in parallel")
                size_box = gr.Textbox(label="Target Size (WxH)", info="Target for output image size", value="512x768")

            # --- Paramètres ---
            with gr.Row():
                resize_checkbox = gr.Checkbox(label="Resize proportionally", value=True)
                crop_checkbox = gr.Checkbox(label="Auto-crop to fit target size", value=True)
                recursive = gr.Checkbox(value=True, label="Recursive")
                delete = gr.Checkbox(value=True, label="Delete old files that have been converted")
                force = gr.Checkbox(value=True, label="Convert files even if it does not contain infotext")

            # --- Bouton de lancement ---
            process_btn = gr.Button("Process", variant="primary")

            # --- Fonction pour récupérer la liste des dossiers ---
            def get_folder_list(folders_text):
                return [f.strip() for f in folders_text.splitlines() if f.strip() and os.path.exists(f.strip())]

            # --- Assignation du bouton Process ---
            process_btn.click(
                fn=lambda text, ext_from, ext_to, cpu, recursive, delete, force, size, resize, crop: __process(
                    get_folder_list(text), ext_from, ext_to, cpu, recursive, delete, force, size, resize, crop
                ),
                inputs=[folders_input, ext_from, ext_to, cpu, recursive, delete, force, size_box, resize_checkbox, crop_checkbox]
            )

        group_a.__exit__()

        # Sauvegarde dans config pour les autres composants
        for comp in [ext_from, ext_to, cpu, recursive, delete, force]:
            comp.do_not_save_to_config = False

    return [(REEE, "QuickConvertImg", "sd-webui-QuickConvertImg")]

# --- Ajouter l’onglet à WebUI ---
on_ui_tabs(QuickConvertImg)
