"""
Rohil Kulshreshtha
February 3, 2026
CS 5330 - PR-CV - Assignment 2

Image Retrieval System - Gradio Web UI
"""

import gradio as gr
import subprocess
import os
from PIL import Image

# Feature types and their corresponding CSV files
FEATURE_TYPES = {
    "Baseline 5x5": ("baseline_5x5", "data/features/baseline_5x5.csv"),
    "Baseline 7x7": ("baseline_7x7", "data/features/baseline_7x7.csv"),
    "Baseline 9x9": ("baseline_9x9", "data/features/baseline_9x9.csv"),
    "Histogram RG-8": ("histogram_rg_8", "data/features/hist_rg_8.csv"),
    "Histogram RG-16": ("histogram_rg_16", "data/features/hist_rg_16.csv"),
    "Histogram RG-16 Smoothed": ("histogram_rg_16_smooth", "data/features/histogram_rg_16_smooth.csv"),
    "Histogram RGB-8": ("histogram_rgb_8", "data/features/hist_rgb_8.csv"),
    "Multi-Histogram RGB-8": ("histogram_multi_rgb_8", "data/features/hist_multi_rgb_8.csv"),
    "Texture + Color (Sobel)": ("texture_color_8", "data/features/texture_color_8.csv"),
    "Texture (Gabor) + Color": ("texture_color_gabor_8", "data/features/texture_color_gabor_8.csv"),
    "Texture (Laws) + Color": ("texture_color_laws_8", "data/features/texture_color_laws_8.csv"),
    "Texture (Fourier) + Color": ("texture_color_fourier_8", "data/features/texture_color_fourier_8.csv"),
    "Texture (CM) + Color": ("texture_color_cm_8", "data/features/texture_color_cm_8.csv"),
    "Deep ResNet18": ("deep_resnet18", "data/features/ResNet18_olym.csv"),
    "Face-Aware RGB-8": ("face_aware_rgb_8", "data/features/face_aware_rgb_8.csv"),
    "Custom: Centered Object": ("custom_centered_object", "data/features/custom_centered_object.csv"),
    "Custom: Blue Sky Scene": ("custom_blue_sky", "data/features/custom_blue_sky.csv"),
}

# All available distance metrics
ALL_DISTANCE_METRICS = [
    "ssd",
    "l1",
    "linf",
    "intersection",
    "multi_intersection",
    "texture_color",
    "texture_color_gabor",
    "texture_color_laws",
    "texture_color_fourier",
    "texture_color_cm",
    "cosine",
    "custom_centered_object",
    "custom_blue_sky",
    "face_aware"
]

# Recommended distance metrics for each feature type
RECOMMENDED_METRICS = {
    "baseline_5x5": ["ssd", "l1", "linf"],
    "baseline_7x7": ["ssd", "l1", "linf"],
    "baseline_9x9": ["ssd", "l1", "linf"],
    "histogram_rg_8": ["intersection"],
    "histogram_rg_16": ["intersection"],
    "histogram_rg_16_smooth": ["intersection"],
    "histogram_rgb_8": ["intersection"],
    "histogram_multi_rgb_8": ["multi_intersection"],
    "texture_color_8": ["texture_color"],
    "texture_color_gabor_8": ["texture_color_gabor"],
    "texture_color_laws_8": ["texture_color_laws"],
    "texture_color_fourier_8": ["texture_color_fourier"],
    "texture_color_cm_8": ["texture_color_cm"],
    "deep_resnet18": ["cosine"],
    "face_aware_rgb_8": ["face_aware"],
    "custom_centered_object": ["custom_centered_object"],
    "custom_blue_sky": ["custom_blue_sky"],
}

"""
# Update distance metric dropdown based on selected feature type.
"""
def update_metric_choices(feature_name):
    feature_type = FEATURE_TYPES[feature_name][0]
    recommended = RECOMMENDED_METRICS[feature_type]
    
    return gr.update(
        choices=ALL_DISTANCE_METRICS,
        value=recommended[0]
    )

"""
Run the C++ match executable and return results.
"""
def find_matches(target_image_path, feature_name, distance_metric, num_matches, match_mode):
    if not target_image_path:
        return [None] * 20 + [gr.update(visible=False)] * 20 + ["❌ Please select a target image"]
    
    feature_type, csv_path = FEATURE_TYPES[feature_name]
    
    # Validate metric compatibility
    recommended = RECOMMENDED_METRICS[feature_type]
    warning = ""
    if distance_metric not in recommended:
        warning = f"⚠️ Warning: '{distance_metric}' may not work well with '{feature_type}'\nRecommended: {', '.join(recommended)}\n\n"
    
    # Build command
    cmd = [
        "./bin/match.exe",
        target_image_path,
        csv_path,
        feature_type,
        distance_metric,
        str(num_matches),
        match_mode.lower()
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse matched filenames
        matches = []
        for line in output.split('\n'):
            if line.strip() and len(line) > 0 and line[0].isdigit() and '. ' in line:
                parts = line.split('. ')
                if len(parts) >= 2:
                    filename = parts[1].split(' (')[0]
                    full_path = os.path.join("data", "olympus", filename)
                    if os.path.exists(full_path):
                        matches.append(full_path)
        
        # Load images and set visibility
        images = []
        visibility_updates = []
        
        for i in range(20):
            if i < len(matches):
                try:
                    img = Image.open(matches[i])
                    img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                    images.append(img)
                    visibility_updates.append(gr.update(visible=True))
                except:
                    images.append(None)
                    visibility_updates.append(gr.update(visible=False))
            else:
                images.append(None)
                visibility_updates.append(gr.update(visible=False))
        
        status = warning + f"✓ Found {len(matches)} matches"
        return images + visibility_updates + [status]
            
    except subprocess.CalledProcessError as e:
        error_lines = [line for line in e.stderr.split('\n') if 'ERROR' in line]
        error_msg = '\n'.join(error_lines) if error_lines else "Match program failed. The distance metric that you selected might not be compatible with the feature type."
        return [None] * 20 + [gr.update(visible=False)] * 20 + [f"❌ Error:\n{error_msg}"]
    except Exception as e:
        return [None] * 20 + [gr.update(visible=False)] * 20 + [f"❌ Error: {str(e)}"]

# Create interface
with gr.Blocks(title="Image Retrieval System", theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🖼️ Content-Based Image Retrieval System")
    gr.Markdown("### CS 5330 - Rohil Kulshreshtha - Assignment 2")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🎯 Search Parameters")
            
            target_image = gr.Image(
                type="filepath", 
                label="Target Image", 
                height=300
            )
            
            feature_type = gr.Dropdown(
                choices=list(FEATURE_TYPES.keys()),
                value=list(FEATURE_TYPES.keys())[0],
                label="Feature Type"
            )
            
            distance_metric = gr.Dropdown(
                choices=ALL_DISTANCE_METRICS,
                value="ssd",
                label="Distance Metric"
            )
            
            num_matches = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Number of Matches"
            )
            
            match_mode = gr.Radio(
                choices=["Top", "Bottom"],
                value="Top",
                label="Match Mode"
            )
            
            search_btn = gr.Button("🔍 Find Matches", variant="primary", size="lg")
            
            status_text = gr.Textbox(
                label="Status",
                lines=4,
                max_lines=8,
                interactive=False
            )
        
        with gr.Column(scale=3):
            gr.Markdown("## 📊 Matching Results")
            
            # Create 5x4 grid using loops
            all_images = []
            
            for row in range(4):
                with gr.Row():
                    for col in range(5):
                        img_num = row * 5 + col + 1
                        img = gr.Image(label=str(img_num), height=180, width=180, visible=False)
                        all_images.append(img)
    
    # Update distance metric when feature type changes
    feature_type.change(
        fn=update_metric_choices,
        inputs=[feature_type],
        outputs=[distance_metric]
    )
    
    # Connect search button
    search_btn.click(
        fn=find_matches,
        inputs=[target_image, feature_type, distance_metric, num_matches, match_mode],
        outputs=all_images + all_images + [status_text]
    )
    
    with gr.Accordion("📖 Instructions", open=False):
        gr.Markdown("""
        ### How to Use:
        1. Select target image from `data/olympus/`
        2. Choose feature type (auto-updates distance metric)
        3. Optionally override distance metric
        4. Set number of matches (1-20)
        5. Choose match mode (Top/Bottom)
        6. Click Find Matches
        
        ### Feature Types:
        - **Baseline**: Raw pixel comparison from center region
        - **Histogram**: Color distribution matching
        - **Multi-Histogram**: Spatial color analysis
        - **Texture + Color**: Combined texture and color features
        - **Deep ResNet18**: Neural network semantic matching
        - **Custom**: Specialized features for specific scenarios
        
        ### Recommended Pairings:
        - Baseline → ssd, l1, linf
        - Histogram → intersection
        - Texture+Color → specific texture metrics
        - Deep ResNet18 → cosine
        - Face Aware → face aware
        - Custom Features → custom distance metrics
        """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860, show_error=True, css="""button[aria-label="Download"], 
                button[aria-label="Share"], button[aria-label="Fullscreen"] {display: none !important;}""")