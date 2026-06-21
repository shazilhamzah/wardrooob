import json
import os

try:
    import garmentiq as giq
    from garmentiq.classification.model_definition import tinyViT
    from garmentiq.landmark.detection.model_definition import PoseHighResolutionNet
    from garmentiq.garment_classes import garment_classes
    from garmentiq.landmark.derivation.derivation_dict import derivation_dict
    from garmentiq.segmentation.model_definition.birefnet import BiRefNet, load_birefnet_config
except ImportError:
    print("Warning: 'garmentiq' module not found. Please ensure it is installed in your environment.")
    # We allow the script to be importable for testing purposes even if garmentiq is missing.
    giq = None
    tinyViT = None
    PoseHighResolutionNet = None
    garment_classes = {}
    derivation_dict = {}
    BiRefNet = None
    
    def load_birefnet_config():
        return {}


def get_tailor_agent(input_dir="../input_images", model_dir="../models", output_dir="../output"):
    """
    Initializes and returns the GarmentIQ tailor agent.
    """
    if giq is None:
        raise ImportError("garmentiq library is required to run the agent.")
        
    # Setup the Tailor Agent
    # This orchestrates classification, segmentation, and measurement in one go.
    tailor_agent = giq.tailor(
        input_dir=input_dir,            # Folder containing the images you want to measure
        model_dir=model_dir,            # Folder where downloaded models are stored
        output_dir=output_dir,          # Folder where results, masks, and JSON will be saved
        class_dict=garment_classes,
        
        # Enable landmark refinement (requires segmentation) and derivation
        do_derive=True,
        derivation_dict=derivation_dict,
        do_refine=True,
        
        # Classification Configuration (identifies the garment type)
        classification_model_path="tiny_vit_inditex_finetuned.pt",
        classification_model_class=tinyViT,
        classification_model_args={
            "num_classes": len(list(garment_classes.keys())) if garment_classes else 0,
            "img_size": (120, 184),
            "patch_size": 6,
            "resize_dim": (120, 184),
            "normalize_mean": [0.8047, 0.7808, 0.7769],
            "normalize_std": [0.2957, 0.3077, 0.3081],
        },
        
        # Segmentation Configuration (removes the background)
        segmentation_model_path="birefnet/model.safetensors",
        segmentation_model_class=BiRefNet,
        segmentation_model_args={
            "model_config": load_birefnet_config(),
            "resize_dim": (1024, 1024),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "background_color": [102, 255, 102] # Replaces background with solid green
        },
        
        # Landmark Detection Configuration (finds the measurement points)
        landmark_detection_model_path="hrnet.pth",
        landmark_detection_model_class=PoseHighResolutionNet(),
        landmark_detection_model_args={
            "scale_std": 200.0,
            "resize_dim": [288, 384],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
    )
    return tailor_agent

def preprocess_images(input_dir):
    """
    Scans the input directory for images and applies EXIF orientation transpositions,
    physically rotating the pixel data and saving it back to disk. This fixes issues
    where images rotated in Windows Photo Viewer still load in their original orientation
    in PIL/OpenCV.
    """
    from PIL import Image, ImageOps
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            image_path = os.path.join(input_dir, filename)
            try:
                with Image.open(image_path) as img:
                    exif = img.getexif()
                    # 274 is the EXIF Orientation tag
                    if exif and 274 in exif:
                        print(f"Applying physical rotation from EXIF metadata for: {filename}")
                        transposed_img = ImageOps.exif_transpose(img)
                        # We save it back to overwrite the file, removing the EXIF orientation
                        # tag so it's not applied twice by other tools.
                        transposed_img.save(image_path, quality=95)
                        transposed_img.close()
            except Exception as e:
                print(f"Warning: Failed to preprocess image orientation for {filename}: {e}")

def main():
    # Paths are relative to the location of this script (stage1/scripts)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(base_dir, "../input_images"))
    model_dir = os.path.abspath(os.path.join(base_dir, "../models"))
    output_dir = os.path.abspath(os.path.join(base_dir, "../output"))
    
    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess image orientations
    preprocess_images(input_dir)
    
    print("Initializing GarmentIQ Tailor Agent...")
    try:
        tailor_agent = get_tailor_agent(input_dir, model_dir, output_dir)
    except ImportError as e:
        print(f"Error: {e}")
        return

    # View Agent Configuration
    tailor_agent.summary()

    # Execute the Pipeline
    # This runs the background cleaning and point mapping, saving visual outputs.
    print(f"\nProcessing images from: {input_dir}")
    metadata, outputs = tailor_agent.measure(
        save_segmentation_image=True, 
        save_measurement_image=True
    )

    # Access and Print the Measurement Results
    print("\n--- Measurement Processing Complete ---")
    print("Access processed files via metadata mapping:", metadata)

    # Print the resulting JSON data for each processed image
    if 'measurement_json' in metadata:
        for json_path in metadata['measurement_json']:
            if os.path.exists(json_path):
                with open(json_path, 'r') as file:
                    data = json.load(file)
                    print(f"\nResults for {json_path}:")
                    print(json.dumps(data, indent=4, sort_keys=True))
            else:
                print(f"Warning: Expected output file not found at {json_path}")
    else:
        print("No 'measurement_json' key found in metadata.")

if __name__ == "__main__":
    main()
