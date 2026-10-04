import os
import tempfile
import xml.etree.ElementTree as ET
import cv2
import re
from data_pipeline.minio_client import SignPoseStorage

def parse_eaf_timestamps(eaf_path):
    """Extracts gloss text and millisecond boundaries from the ELAN file."""
    tree = ET.parse(eaf_path)
    root = tree.getroot()
    
    time_slots = {ts.attrib['TIME_SLOT_ID']: int(ts.attrib['TIME_VALUE']) 
                  for ts in root.iter('TIME_SLOT')}
    
    glosses = []
    for tier in root.iter('TIER'):
        if tier.attrib.get('TIER_ID') in ['Glossa mà activa', 'Glossa mà passiva']:
            for ann in tier.iter('ALIGNABLE_ANNOTATION'):
                start_ms = time_slots[ann.attrib['TIME_SLOT_REF1']]
                end_ms = time_slots[ann.attrib['TIME_SLOT_REF2']]
                val = ann.find('ANNOTATION_VALUE').text
                if val:
                    glosses.append((val, start_ms, end_ms))
    return glosses

def slice_video_clip(input_video, output_mp4, start_ms, end_ms):
    """Uses OpenCV to extract a specific temporal window from a video."""
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int((start_ms / 1000.0) * fps)
    end_frame = int((end_ms / 1000.0) * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
    
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        
    cap.release()
    out.release()

def sanitize_filename(text):
    """Removes special characters and standardizes case to prevent duplicates."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text).upper()

def process_raw_dataset():
    storage = SignPoseStorage(endpoint="minio:9000")
    raw_bucket = "raw-data"
    raw_prefix = "lsc-upf/"
    vocab_bucket = "vocabulary-videos"
    
    seen_glosses = set()
    stats = {
        "total_eaf_processed": 0,
        "successful_clips": 0,
        "duplicate_skips": 0,
        "failed_missing_video": 0
    }
    
    # Isolate .eaf files within the specific directory
    all_eaf_objects = storage.list_objects(raw_bucket, suffix=".eaf")
    target_eaf_objects = [obj for obj in all_eaf_objects if obj.startswith(raw_prefix)]
    
    # Pre-fetch video paths for O(1) lookup to support multiple formats
    all_mp4 = storage.list_objects(raw_bucket, suffix=".mp4")
    all_mov = storage.list_objects(raw_bucket, suffix=".mov")
    all_videos = set(all_mp4 + all_mov)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for eaf_obj in target_eaf_objects:
            stats["total_eaf_processed"] += 1
            base_path = eaf_obj.rsplit('.', 1)[0]
            
            mp4_obj = base_path + '.mp4'
            mov_obj = base_path + '.mov'
            
            # Determine which video format exists in MinIO
            target_video_obj = None
            if mp4_obj in all_videos:
                target_video_obj = mp4_obj
            elif mov_obj in all_videos:
                target_video_obj = mov_obj
                
            if not target_video_obj:
                print(f"Skipping {eaf_obj}: Missing corresponding .mp4 or .mov in MinIO.")
                stats["failed_missing_video"] += 1
                continue
                
            local_eaf = os.path.join(temp_dir, "temp.eaf")
            # Preserve the original video extension for OpenCV demuxing
            local_video = os.path.join(temp_dir, f"temp.{target_video_obj.split('.')[-1]}")
            
            storage.download_file(raw_bucket, eaf_obj, local_eaf)
            storage.download_file(raw_bucket, target_video_obj, local_video)
                
            gloss_data = parse_eaf_timestamps(local_eaf)
            
            for gloss, start_ms, end_ms in gloss_data:
                clean_gloss = sanitize_filename(gloss)
                
                # Strict deduplication check
                if clean_gloss in seen_glosses:
                    stats["duplicate_skips"] += 1
                    continue
                
                base_name = eaf_obj.split('/')[-1].rsplit('.', 1)[0]
                output_name = f"{clean_gloss}/{clean_gloss}_{base_name}.mp4"
                
                local_clip = os.path.join(temp_dir, "clip.mp4")
                slice_video_clip(local_video, local_clip, start_ms, end_ms)
                
                # Upload the isolated clip to the vocabulary bucket
                storage.upload_file(vocab_bucket, output_name, local_clip)
                
                seen_glosses.add(clean_gloss)
                stats["successful_clips"] += 1
                
                if os.path.exists(local_clip):
                    os.remove(local_clip)
                    
    print("\n--- Extraction Pipeline Statistics ---")
    print(f"Total .eaf files processed:       {stats['total_eaf_processed']}")
    print(f"Distinct vocabulary items saved:  {stats['successful_clips']}")
    print(f"Duplicate glosses skipped:        {stats['duplicate_skips']}")
    print(f"Failed pairs (missing video):     {stats['failed_missing_video']}")
    print("--------------------------------------")

if __name__ == "__main__":
    process_raw_dataset()