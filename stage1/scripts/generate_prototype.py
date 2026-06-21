# Monkeypatch for legacy libraries (like chumpy) compatibility with Python 3.11+
import inspect
if not hasattr(inspect, 'getargspec'):
    import collections
    ArgSpec = collections.namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])
    def getargspec(func):
        args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(func)
        return ArgSpec(args, varargs, varkw, defaults)
    inspect.getargspec = getargspec

import collections
import collections.abc
for name in ['Mapping', 'MutableMapping', 'Sequence', 'MutableSequence', 'Iterable', 'Callable']:
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))

import numpy as np
# Add legacy numpy type aliases removed in numpy 2.0+
for name, target in [('bool', bool), ('int', int), ('float', float), ('complex', complex), ('object', object), ('unicode', str), ('str', str)]:
    if not hasattr(np, name):
        setattr(np, name, target)

import os
import json
import torch
import smplx
import trimesh
import pyrender
from pathlib import Path
import math

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_smpl_model(model_dir):
    """
    Returns smplx body model object.
    Requires SMPL_NEUTRAL.pkl to be in model_dir/smpl.
    """
    smpl_folder = os.path.join(model_dir, 'smpl')
    
    try:
        model = smplx.create(smpl_folder, model_type='smpl',
                             gender='neutral', use_face_contour=False,
                             num_betas=300, ext='pkl')
        print("Loaded SMPL model with 300 shape components.")
        return model, 300
    except Exception as e:
        print(f"Failed to load with 300 betas ({e}). Falling back to 10 betas.")
        try:
            model = smplx.create(smpl_folder, model_type='smpl',
                                 gender='neutral', use_face_contour=False,
                                 num_betas=10, ext='pkl')
            return model, 10
        except Exception as e2:
            print(f"Error loading SMPL model from {smpl_folder}: {e2}")
            print("Please ensure you have downloaded SMPL_NEUTRAL.pkl from the SMPL website and placed it in the models/smpl directory.")
            raise

def optimize_betas_for_measurements(smpl_model, target_measurements, num_betas):
    """
    Optimizes SMPL betas to match target measurements using PyTorch Adam optimizer.
    """
    print("Initializing optimization to match exact measurements...")
    # Get neutral vertices to find measurement loops dynamically
    neutral_output = smpl_model(betas=torch.zeros(1, num_betas))
    neutral_v = neutral_output.vertices.detach()
    
    max_y = torch.max(neutral_v[0, :, 1]).item()
    min_y = torch.min(neutral_v[0, :, 1]).item()
    H = max_y - min_y
    
    # Proportional heights for chest, waist, hips
    chest_y = max_y - H * 0.26
    waist_y = max_y - H * 0.35
    hip_y = max_y - H * 0.48
    
    def get_loop(y_target):
        # Mask out arms by restricting X coordinate
        mask = (neutral_v[0, :, 1] > y_target - 0.02) & (neutral_v[0, :, 1] < y_target + 0.02) & (neutral_v[0, :, 0] > -0.18) & (neutral_v[0, :, 0] < 0.18)
        indices = torch.nonzero(mask).squeeze()
        if len(indices) > 0:
            pts = neutral_v[0, indices]
            center = torch.mean(pts, dim=0)
            angles = torch.atan2(pts[:, 2] - center[2], pts[:, 0] - center[0])
            sorted_idx = torch.argsort(angles)
            return indices[sorted_idx]
        return indices

    chest_idx = get_loop(chest_y)
    waist_idx = get_loop(waist_y)
    hip_idx = get_loop(hip_y)
    
    def calc_meas(v):
        h = (torch.max(v[0, :, 1]) - torch.min(v[0, :, 1])) * 100.0
        def circ(idx):
            if len(idx) < 3: return torch.tensor(0.0, device=v.device)
            pts = v[0, idx]
            pts_shifted = torch.roll(pts, shifts=-1, dims=0)
            return torch.sum(torch.norm(pts - pts_shifted, dim=1)) * 100.0
        return h, circ(chest_idx), circ(waist_idx), circ(hip_idx)

    # Initialize betas as trainable
    betas = torch.zeros(1, num_betas, requires_grad=True)
    optimizer = torch.optim.Adam([betas], lr=0.5)
    
    target_h = target_measurements.get('height_cm', 175)
    target_c = target_measurements.get('chest_cm', 95)
    target_w = target_measurements.get('waist_cm', 80)
    target_p = target_measurements.get('hip_cm', 98)
    
    print(f"Target Measurements: Height {target_h}, Chest {target_c}, Waist {target_w}, Hips {target_p}")
    
    for i in range(400):
        optimizer.zero_grad()
        output = smpl_model(betas=betas)
        v = output.vertices
        
        h, c, w, p = calc_meas(v)
        
        loss_h = (h - target_h)**2
        loss_c = (c - target_c)**2
        loss_w = (w - target_w)**2
        loss_p = (p - target_p)**2
        
        # Regularization to keep shape realistic (penalize higher PCs more)
        weights = torch.linspace(1.0, 10.0, num_betas).to(betas.device)
        loss_reg = torch.sum((betas ** 2) * weights) * 0.01
        
        loss = loss_h + loss_c + loss_w + loss_p + loss_reg
        loss.backward()
        optimizer.step()
        
        if (i+1) % 50 == 0:
            print(f"Step {i+1}: Loss {loss.item():.2f} | H:{h.item():.1f} C:{c.item():.1f} W:{w.item():.1f} Hip:{p.item():.1f}")
            
    print("Optimization finished.")
    return betas.detach()

def apply_skin_tone(mesh, rgb_tuple):
    """
    Modifies vertex colors of the trimesh.
    """
    color = [rgb_tuple[0], rgb_tuple[1], rgb_tuple[2], 255]
    mesh.visual.vertex_colors = color
    return mesh

def generate_default_hair(body_vertices, output_path):
    """
    Generates a simple default hair cap mesh based on the body's head position
    and saves it to output_path.
    """
    # Head vertices are at the very top of the body mesh
    y_coords = body_vertices[:, 1]
    max_y = np.max(y_coords)
    
    # Take the top 20cm of the body mesh
    head_indices = np.where(y_coords > (max_y - 0.20))[0]
    head_vertices = body_vertices[head_indices]
    
    # Calculate head center
    head_center = np.mean(head_vertices, axis=0)
    
    # Create a sphere mesh to represent a simple hair cap
    hair_sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.095)
    
    # Shift the sphere slightly up and back to sit nicely on the scalp
    translation = head_center + np.array([0.0, 0.03, -0.01])
    hair_sphere.apply_translation(translation)
    
    # Scale it slightly to look more like a hair cap
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = 1.08  # Width
    scale_matrix[1, 1] = 0.95  # Height
    scale_matrix[2, 2] = 1.12  # Depth
    
    # Apply scaling relative to translation center
    hair_sphere.apply_translation(-translation)
    hair_sphere.apply_transform(scale_matrix)
    hair_sphere.apply_translation(translation)
    
    # Ensure directory exists and export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hair_sphere.export(output_path)
    print(f"Generated default hair model at {output_path}")

def overlay_hair(mesh_scene, hairstyle_id, hair_rgb, model_dir):
    """
    Adds hair mesh tinted to color.
    """
    hair_path = os.path.join(model_dir, 'hair', f"{hairstyle_id}.obj")
    
    if not os.path.exists(hair_path):
        print(f"Warning: Hair model not found at {hair_path}. Skipping hair overlay.")
        return mesh_scene

    try:
        hair_mesh = trimesh.load(hair_path, force='mesh')
        
        color = [hair_rgb[0], hair_rgb[1], hair_rgb[2], 255]
        hair_mesh.visual.vertex_colors = color
        
        pyrender_hair = pyrender.Mesh.from_trimesh(hair_mesh, smooth=True)
        mesh_scene.add(pyrender_hair)
    except Exception as e:
        print(f"Failed to load or overlay hair: {e}")
        
    return mesh_scene

def get_apose_body_pose():
    """
    Returns a body_pose tensor for a natural A-pose.
    Rotates the shoulders slightly downward so the arms hang naturally
    instead of the stiff T-pose.
    """
    body_pose = torch.zeros(1, 69)
    # Joint 15 = left shoulder  (indices 45-47)
    # Joint 16 = right shoulder (indices 48-50)
    body_pose[0, 47] =  0.45   # left  shoulder: rotate arm down
    body_pose[0, 50] = -0.45   # right shoulder: rotate arm down
    # Slight elbow bend for natural look
    body_pose[0, 56] =  0.15   # left  elbow
    body_pose[0, 59] = -0.15   # right elbow
    return body_pose


def render_front_view(scene, output_path, body_min_y=0.0, body_max_y=1.7):
    """
    Renders a premium quality front-view portrait of the body with:
    - Dynamic camera framing based on actual body height
    - 3-point lighting (key + fill + rim)
    - Gradient background
    - Portrait aspect ratio
    """
    from PIL import Image, ImageDraw, ImageFilter

    body_height  = body_max_y - body_min_y
    body_center_y = (body_max_y + body_min_y) / 2.0

    # ── Camera ────────────────────────────────────────────────────────────────
    # Narrow FoV avoids perspective distortion (telephoto look)
    fov = np.pi / 5.0        # ~36 degrees
    # Distance needed to see full body + 20% margin
    margin = body_height * 0.20
    cam_z  = ((body_height / 2.0 + margin) / np.tan(fov / 2.0))

    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=0.667)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, body_center_y],
        [0.0, 0.0, 1.0, cam_z],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    # ── 3-Point Lighting ──────────────────────────────────────────────────────
    # Key light – warm, upper-right front
    key = pyrender.DirectionalLight(color=np.array([1.0, 0.95, 0.87]), intensity=5.0)
    key_pose = np.eye(4)
    key_pose[:3, 3] = [0.8, body_center_y + body_height * 0.3, cam_z - 0.4]
    scene.add(key, pose=key_pose)

    # Fill light – cool, upper-left, softer
    fill = pyrender.DirectionalLight(color=np.array([0.8, 0.88, 1.0]), intensity=2.5)
    fill_pose = np.eye(4)
    fill_pose[:3, 3] = [-0.8, body_center_y + body_height * 0.2, cam_z - 0.4]
    scene.add(fill, pose=fill_pose)

    # Rim/back light – pure white, from behind the body for edge definition
    rim = pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]), intensity=1.8)
    rim_pose = np.eye(4)
    rim_pose[:3, 3] = [0.0, body_center_y + body_height * 0.4, -1.5]
    scene.add(rim, pose=rim_pose)

    # ── Render ────────────────────────────────────────────────────────────────
    vp_w, vp_h = 768, 1152          # portrait: taller than wide
    r = pyrender.OffscreenRenderer(viewport_width=vp_w, viewport_height=vp_h)
    color, _ = r.render(scene)
    r.delete()

    # ── Gradient background composite ─────────────────────────────────────────
    bg = Image.new('RGB', (vp_w, vp_h))
    draw = ImageDraw.Draw(bg)
    for y_px in range(vp_h):
        t = y_px / vp_h
        # Soft blue-grey gradient: lighter at top, slightly warmer at bottom
        rv = int(200 + (225 - 200) * t)
        gv = int(210 + (230 - 210) * t)
        bv = int(225 + (240 - 225) * t)
        draw.line([(0, y_px), (vp_w, y_px)], fill=(rv, gv, bv))

    # Mask out the white pyrender background, soften edges by 1px
    body_img  = Image.fromarray(color)
    mask_arr  = np.any(color != [255, 255, 255], axis=-1).astype(np.uint8) * 255
    mask_img  = Image.fromarray(mask_arr, 'L').filter(ImageFilter.GaussianBlur(radius=1))
    bg.paste(body_img, (0, 0), mask_img)

    bg.save(output_path)
    print(f"Saved prototype body render to {output_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(base_dir, "../models"))
    body_models_dir = os.path.abspath(os.path.join(base_dir, "../body_models"))
    config_path = os.path.join(body_models_dir, "body_config.json")
    output_path = os.path.join(body_models_dir, "prototype_body.png")
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return

    config = load_config(config_path)
    
    print("Loading SMPL model...")
    try:
        smpl_model, num_betas = load_smpl_model(model_dir)
    except Exception:
        return

    print("Optimizing betas from exact measurements...")
    betas = optimize_betas_for_measurements(smpl_model, config, num_betas)

    print("Generating body mesh (A-pose)...")
    body_pose = get_apose_body_pose()
    output = smpl_model(betas=betas, body_pose=body_pose)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = smpl_model.faces

    # Compute proper normals for smooth shading
    body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    body_mesh.fix_normals()
    skin_rgb = config.get("skin_rgb", [224, 172, 105])
    body_mesh = apply_skin_tone(body_mesh, skin_rgb)

    # Improve ambient light for softer base illumination
    scene = pyrender.Scene(ambient_light=[0.25, 0.22, 0.20], bg_color=[255, 255, 255, 0])

    pyrender_body = pyrender.Mesh.from_trimesh(body_mesh, smooth=True)
    scene.add(pyrender_body)

    hairstyle_id = config.get("hairstyle_id")
    hair_rgb = config.get("hair_color_rgb", [0, 0, 0])
    if hairstyle_id:
        hair_path = os.path.join(model_dir, 'hair', f"{hairstyle_id}.obj")
        if not os.path.exists(hair_path):
            generate_default_hair(vertices, hair_path)
        scene = overlay_hair(scene, hairstyle_id, hair_rgb, model_dir)

    # Pass actual body bounds for dynamic camera framing
    body_min_y = float(vertices[:, 1].min())
    body_max_y = float(vertices[:, 1].max())

    print("Rendering scene...")
    render_front_view(scene, output_path, body_min_y=body_min_y, body_max_y=body_max_y)

if __name__ == "__main__":
    main()
