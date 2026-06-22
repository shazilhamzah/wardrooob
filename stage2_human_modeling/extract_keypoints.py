import cv2
import mediapipe as mp
import json
import os
import glob
import numpy as np

def create_body25_from_mp(pose_landmarks, img_w, img_h):
    if not pose_landmarks:
        return [0.0] * 75
        
    def get_pt(idx):
        landmark = pose_landmarks.landmark[idx]
        return [landmark.x * img_w, landmark.y * img_h, landmark.visibility]
        
    pts = [get_pt(i) for i in range(33)]
    
    body25 = [[0.0, 0.0, 0.0]] * 25
    
    # 0: Nose
    body25[0] = pts[0]
    # 1: Neck (midpoint of shoulders)
    body25[1] = [
        (pts[11][0] + pts[12][0]) / 2,
        (pts[11][1] + pts[12][1]) / 2,
        (pts[11][2] + pts[12][2]) / 2
    ]
    # 2: RShoulder
    body25[2] = pts[12]
    # 3: RElbow
    body25[3] = pts[14]
    # 4: RWrist
    body25[4] = pts[16]
    # 5: LShoulder
    body25[5] = pts[11]
    # 6: LElbow
    body25[6] = pts[13]
    # 7: LWrist
    body25[7] = pts[15]
    # 8: MidHip
    body25[8] = [
        (pts[23][0] + pts[24][0]) / 2,
        (pts[23][1] + pts[24][1]) / 2,
        (pts[23][2] + pts[24][2]) / 2
    ]
    # 9: RHip
    body25[9] = pts[24]
    # 10: RKnee
    body25[10] = pts[26]
    # 11: RAnkle
    body25[11] = pts[28]
    # 12: LHip
    body25[12] = pts[23]
    # 13: LKnee
    body25[13] = pts[25]
    # 14: LAnkle
    body25[14] = pts[27]
    # 15: REye
    body25[15] = pts[5]
    # 16: LEye
    body25[16] = pts[2]
    # 17: REar
    body25[17] = pts[8]
    # 18: LEar
    body25[18] = pts[7]
    # 19: LBigToe
    body25[19] = pts[31]
    # 20: LSmallToe (Not perfectly mapped, use 0 to ignore)
    body25[20] = [0.0, 0.0, 0.0]
    # 21: LHeel
    body25[21] = pts[29]
    # 22: RBigToe
    body25[22] = pts[32]
    # 23: RSmallToe (Not perfectly mapped, use 0 to ignore)
    body25[23] = [0.0, 0.0, 0.0]
    # 24: RHeel
    body25[24] = pts[30]
    
    return [val for pt in body25 for val in pt]

def create_hand_from_mp(hand_landmarks, img_w, img_h):
    if not hand_landmarks:
        return [0.0] * 63
    hand_pts = []
    for lm in hand_landmarks.landmark:
        hand_pts.extend([lm.x * img_w, lm.y * img_h, 1.0])
    return hand_pts

def process_images():
    img_folder = 'samples/images'
    out_folder = 'samples/openpose'
    os.makedirs(out_folder, exist_ok=True)
    
    mp_holistic = mp.solutions.holistic
    
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=False) as holistic:
        
        for img_path in glob.glob(os.path.join(img_folder, '*.*')):
            basename = os.path.basename(img_path)
            name, _ = os.path.splitext(basename)
            out_json = os.path.join(out_folder, f'{name}_keypoints.json')
            out_json2 = os.path.join(out_folder, f'{name}.json')
            
            if os.path.exists(out_json) or os.path.exists(out_json2):
                continue
                
            print(f"Extracting keypoints for {img_path}...")
            image = cv2.imread(img_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            img_h, img_w, _ = image.shape
            
            body_25_flat = create_body25_from_mp(results.pose_landmarks, img_w, img_h)
            left_hand_flat = create_hand_from_mp(results.left_hand_landmarks, img_w, img_h)
            right_hand_flat = create_hand_from_mp(results.right_hand_landmarks, img_w, img_h)
            
            # 70 face points (210 values)
            face_flat = [0.0] * 210
            
            person_data = {
                "person_id": [-1],
                "pose_keypoints_2d": body_25_flat,
                "face_keypoints_2d": face_flat,
                "hand_left_keypoints_2d": left_hand_flat,
                "hand_right_keypoints_2d": right_hand_flat,
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }
            
            json_data = {
                "version": 1.3,
                "people": [person_data]
            }
            
            with open(out_json, 'w') as f:
                json.dump(json_data, f)
                
if __name__ == "__main__":
    process_images()
