import os
import shutil
import json
import glob


os.makedirs(os.path.join(output_dir, 'vision/data/jpg/jpg'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'vision/docs'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'vision/data/video_shot_boundaries/representative_frames'), exist_ok=True)
with open(os.path.join(output_dir, 'vision/docs/video_data.msb'), 'w') as writer:
    writer.write('')
with open(os.path.join(output_dir, 'vision/docs/masterShotBoundary.msb'), 'w') as writer:
    writer.write('')
with open(os.path.join(output_dir, 'vision/docs/parent_children.tab'), 'w') as writer:
    for img_id in image_list:
        writer.write('gaia\t0\t%s\t%s\t0\t.ltf.xml\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n' % (
            img_id, img_id
        ))
    for caption_id in caption_list:
        img_id = img_caption_mapping[caption_id]
        writer.write('gaia\t0\t%s\t%s\t0\t.ltf.xml\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n' % (
            img_id, caption_id
        ))