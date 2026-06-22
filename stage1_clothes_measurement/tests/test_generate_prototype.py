import os
import sys
import json
import pytest
from PIL import Image
import numpy as np

# Add scripts directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from generate_prototype import main

def test_generate_prototype_output():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(base_dir, "../models"))
    smpl_path = os.path.join(model_dir, "smpl", "SMPL_NEUTRAL.pkl")
    
    if not os.path.exists(smpl_path):
        pytest.skip(f"SMPL model not found at {smpl_path}. Skipping integration test.")
        
    body_models_dir = os.path.abspath(os.path.join(base_dir, "../body_models"))
    output_path = os.path.join(body_models_dir, "prototype_body.png")
    config_path = os.path.join(body_models_dir, "body_config.json")
    
    # Ensure config exists
    assert os.path.exists(config_path), "Config file must exist."
    
    # Remove existing output
    if os.path.exists(output_path):
        os.remove(output_path)
        
    # Run pipeline
    main()
    
    # Assert output exists
    assert os.path.exists(output_path), "Output image was not generated."
    
    # Assert correct resolution and non-blank
    img = Image.open(output_path).convert('RGB')
    assert img.size == (1024, 1024), "Output image resolution is incorrect."
    
    img_np = np.array(img)
    std_dev = np.std(img_np)
    assert std_dev > 10.0, "Output image appears to be blank."
    
    # Assert skin tone average roughly matches (excluding white background)
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    target_rgb = config.get("skin_rgb", [224, 172, 105])
    
    # Find non-background pixels (assuming bg is white or transparent, here white from rendering)
    # Background might be [255, 255, 255]
    mask = np.any(img_np != [255, 255, 255], axis=-1)
    if np.sum(mask) > 0:
        avg_color = np.mean(img_np[mask], axis=0)
        
        # Tolerance is large because rendering adds lighting/shading
        diff = np.abs(avg_color - target_rgb)
        assert np.mean(diff) < 60, f"Average color {avg_color} is too far from target {target_rgb}."
