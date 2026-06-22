import os
import json
import argparse

def get_default_reference_key(predicted_keys):
    if "front length" in predicted_keys:
        return "front length"
    if "full length" in predicted_keys:
        return "full length"
    if predicted_keys:
        return sorted(list(predicted_keys))[0]
    return None

def evaluate_file(json_path, actual_dir, custom_scale_factor=None, unit="inches"):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get the image name
    img_key = list(data.keys())[0]
    results = data[img_key]
    predictions = results.get("measurements", {})
    garment_class = results.get('class', 'unknown')
    
    # Resolve the actual measurement file path
    base_name = os.path.splitext(img_key)[0]
    actual_file_path = os.path.join(actual_dir, f"{base_name}.json")
    
    # If actual file doesn't exist, create a template
    if not os.path.exists(actual_file_path):
        template = {key: None for key in predictions.keys()}
        try:
            with open(actual_file_path, 'w') as f_out:
                json.dump(template, f_out, indent=2)
            print(f"Created template actual measurement file at: {actual_file_path}")
            print(f"Please fill in the physical measurements for this image to compute accuracy.")
        except Exception as e:
            print(f"Warning: Could not create template file: {e}")
        return None
        
    with open(actual_file_path, 'r') as f_actual:
        try:
            actual_measurements = json.load(f_actual)
        except Exception as e:
            print(f"Error reading actual measurement file {actual_file_path}: {e}")
            return None
            
    # Check if actual measurements has any non-null and positive values
    valid_actuals = {k: v for k, v in actual_measurements.items() if v is not None and v > 0}
    if not valid_actuals:
        print(f"Skipping {img_key}: No actual measurements filled in '{actual_file_path}'.")
        return None

    # Determine calibration
    reference_key = get_default_reference_key(predictions.keys())
    
    print(f"\n========================================================")
    print(f"GarmentIQ Accuracy Evaluator")
    print(f"========================================================")
    print(f"Image: {img_key}")
    print(f"Garment Class: {garment_class}")
    print(f"Units: {unit}")
    print(f"--------------------------------------------------------")
    
    # Determine the scale factor (units per pixel)
    if custom_scale_factor is not None:
        scale_factor = custom_scale_factor
        print(f"Using custom scale factor: {scale_factor:.6f} {unit}/pixel")
    else:
        if reference_key not in predictions:
            print(f"Error: Reference key '{reference_key}' not found in predictions. Available keys: {list(predictions.keys())}")
            return None
        if reference_key not in actual_measurements or actual_measurements[reference_key] is None or actual_measurements[reference_key] <= 0:
            # Look for any other valid actual measurement key to use as reference
            fallback_ref = None
            for key in predictions.keys():
                if key in actual_measurements and actual_measurements[key] is not None and actual_measurements[key] > 0:
                    fallback_ref = key
                    break
            
            if fallback_ref is None:
                print(f"Error: No valid actual measurements to calibrate against in '{actual_file_path}'.")
                return None
            
            reference_key = fallback_ref
            
        pixel_ref_dist = predictions[reference_key]["distance"]
        actual_ref_val = actual_measurements[reference_key]
        
        # Calculate scale factor (units per pixel)
        scale_factor = actual_ref_val / pixel_ref_dist
        print(f"Calibrated using reference: '{reference_key}'")
        print(f"  Pixel distance: {pixel_ref_dist:.2f} px")
        print(f"  Actual size: {actual_ref_val:.2f} {unit}")
        print(f"  Scale factor: {scale_factor:.6f} {unit}/pixel")
    
    print(f"--------------------------------------------------------")
    print(f"{'Measurement':<18} | {'Pixel Dist':<10} | {f'Pred ({unit})':<11} | {f'Act ({unit})':<10} | {'Abs Error':<9} | {'Error %':<8}")
    print(f"--------------------------------------------------------")
    
    errors = []
    for name, details in predictions.items():
        pixel_dist = details["distance"]
        pred_val = pixel_dist * scale_factor
        
        actual_val = actual_measurements.get(name)
        if actual_val is not None and actual_val > 0:
            abs_err = abs(actual_val - pred_val)
            pct_err = (abs_err / actual_val) * 100
            errors.append(pct_err)
            print(f"{name:<18} | {pixel_dist:<10.2f} | {pred_val:<11.2f} | {actual_val:<10.2f} | {abs_err:<9.2f} | {pct_err:<7.2f}%")
        else:
            print(f"{name:<18} | {pixel_dist:<10.2f} | {pred_val:<11.2f} | {'N/A':<10} | {'N/A':<9} | {'N/A':<8}")
            
    print(f"--------------------------------------------------------")
    mape = None
    if errors:
        # Overall MAPE (all keys)
        mape = sum(errors) / len(errors)
        
        # MAPE excluding the calibration reference key
        non_ref_errors = [e for name, e in zip(predictions.keys(), errors) if name != reference_key]
        if non_ref_errors and custom_scale_factor is None:
            mape_non_ref = sum(non_ref_errors) / len(non_ref_errors)
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print(f"MAPE (excluding calibration reference '{reference_key}'): {mape_non_ref:.2f}%")
        else:
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"========================================================\n")
    
    return {
        "image": img_key,
        "class": garment_class,
        "mape": mape,
        "scale_factor": scale_factor
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate accuracy of GarmentIQ predictions.")
    parser.add_argument("--json_dir", default="../output/measurement_json", help="Path to predictions JSON directory")
    parser.add_argument("--actual_dir", default="../actual_measurements", help="Path to actual measurements JSON directory")
    parser.add_argument("--scale", type=float, default=None, help="Optional custom scale factor")
    parser.add_argument("--unit", default="inches", help="Unit of measurement (inches or cm)")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.abspath(os.path.join(base_dir, args.json_dir))
    actual_dir = os.path.abspath(os.path.join(base_dir, args.actual_dir))
    
    if not os.path.exists(json_dir):
        print(f"Error: Predictions directory '{json_dir}' does not exist.")
        return
        
    os.makedirs(actual_dir, exist_ok=True)
    
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
    
    if not json_files:
        print(f"No prediction JSON files found in '{json_dir}'.")
        return
        
    print(f"Scanning '{json_dir}' for prediction outputs...")
    results = []
    
    for json_file in json_files:
        res = evaluate_file(json_file, actual_dir, custom_scale_factor=args.scale, unit=args.unit)
        if res:
            results.append(res)
            
    if results:
        print("\n" + "=" * 56)
        print("OVERALL ACCURACY SUMMARY REPORT".center(56))
        print("=" * 56)
        print(f"{'Image Name':<30} | {'Garment Class':<15} | {'MAPE':<8}")
        print("-" * 56)
        
        valid_mapes = []
        for r in results:
            if r['mape'] is not None:
                print(f"{r['image']:<30} | {r['class']:<15} | {r['mape']:>6.2f}%")
                valid_mapes.append(r['mape'])
            else:
                print(f"{r['image']:<30} | {r['class']:<15} | {'N/A':>7}")
        print("-" * 56)
        if valid_mapes:
            overall_mape = sum(valid_mapes) / len(valid_mapes)
            print(f"{'Overall Average MAPE':<48} : {overall_mape:.2f}%")
        print("=" * 56 + "\n")

if __name__ == "__main__":
    main()
