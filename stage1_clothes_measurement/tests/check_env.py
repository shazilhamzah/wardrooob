libraries = [
    ("opencv", "cv2"),
    ("rembg", "rembg"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
    ("PIL", "PIL"),
    ("trimesh", "trimesh"),
]

for name, module in libraries:
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "unknown")
        print(f"[OK]   {name:<10} version={version}")
    except BaseException as e:  # catches SystemExit too
        print(f"[FAIL] {name:<10} {type(e).__name__}: {e}")