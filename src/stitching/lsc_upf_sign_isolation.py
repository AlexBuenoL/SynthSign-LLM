import xml.etree.ElementTree as ET
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

def parse_eaf_timestamps(eaf_path):
    tree = ET.parse(eaf_path)
    root = tree.getroot()
    
    # Map time slot IDs to their millisecond values
    time_slots = {ts.attrib['TIME_SLOT_ID']: int(ts.attrib['TIME_VALUE']) 
                  for ts in root.iter('TIME_SLOT')}
    
    glosses = []
    # Extract glosses from both active and passive hands
    for tier in root.iter('TIER'):
        if tier.attrib.get('TIER_ID') in ['Glossa mà activa', 'Glossa mà passiva']:
            for ann in tier.iter('ALIGNABLE_ANNOTATION'):
                start_ms = time_slots[ann.attrib['TIME_SLOT_REF1']]
                end_ms = time_slots[ann.attrib['TIME_SLOT_REF2']]
                val = ann.find('ANNOTATION_VALUE').text
                if val:
                    glosses.append((val, start_ms, end_ms))
    return glosses

def process_video_segment(video_path, start_ms, end_ms, holistic):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = int((start_ms / 1000.0) * fps)
    end_frame = int((end_ms / 1000.0) * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    sequence_coordinates = []
    
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        # Note: Implement the 152-dimensional feature extraction here 
        # to isolate the specific face, hand, and pose landmarks.
        if results.pose_landmarks:
            # Placeholder for the flattened array
            sequence_coordinates.append(np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])) 
            
    cap.release()
    return sequence_coordinates

def build_vocabulary(root_dir):
    root_path = Path(root_dir)
    mp_holistic = mp.solutions.holistic
    
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        # Recursively find all .eaf files across the unstructured folders
        for eaf_file in root_path.rglob('*.eaf'):
            mp4_file = eaf_file.with_suffix('.mp4')
            
            if not mp4_file.exists():
                continue
                
            gloss_data = parse_eaf_timestamps(eaf_file)
            
            for gloss, start_ms, end_ms in gloss_data:
                coords = process_video_segment(mp4_file, start_ms, end_ms, holistic)
                
                if coords:
                    # Clean up the gloss string for safe filenames
                    clean_gloss = "".join([c for c in gloss if c.isalpha() or c.isdigit()]).rstrip()
                    save_path = root_path / 'vocabulary' / f"{clean_gloss}_{eaf_file.stem}.npy"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(save_path, coords)