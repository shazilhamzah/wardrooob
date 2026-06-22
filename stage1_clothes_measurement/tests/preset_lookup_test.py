import os
import sys

# Add scripts directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from generate_prototype import measurements_to_betas

def test_preset_lookup():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    smpl_path = os.path.abspath(os.path.join(base_dir, "../models", "smpl", "SMPL_NEUTRAL.pkl"))
    if not os.path.exists(smpl_path):
        import pytest
        pytest.skip(f"SMPL model not found at {smpl_path}. Skipping preset lookup test.")

    print("=" * 60)
    print("Preset Lookup Test: Manual Sanity Review")
    print("=" * 60)
    
    sample_measurements = [
        {"desc": "Average Male", "height_cm": 176, "chest_cm": 98, "waist_cm": 82, "hip_cm": 99},
        {"desc": "Short & Curvy", "height_cm": 158, "chest_cm": 92, "waist_cm": 76, "hip_cm": 102},
        {"desc": "Tall & Thin", "height_cm": 182, "chest_cm": 88, "waist_cm": 74, "hip_cm": 94},
        {"desc": "Large/Tall", "height_cm": 188, "chest_cm": 108, "waist_cm": 95, "hip_cm": 108},
        {"desc": "Small/Thin", "height_cm": 162, "chest_cm": 84, "waist_cm": 68, "hip_cm": 88},
    ]
    
    for sample in sample_measurements:
        print(f"\nTesting: {sample['desc']}")
        print(f"Input: Height {sample['height_cm']}cm, Chest {sample['chest_cm']}cm, Waist {sample['waist_cm']}cm, Hip {sample['hip_cm']}cm")
        betas = measurements_to_betas(sample)
        print(f"Resulting Betas: {betas.numpy().tolist()[0]}")
        
    print("\n" + "=" * 60)
    
if __name__ == "__main__":
    test_preset_lookup()
