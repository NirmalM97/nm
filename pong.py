import os
import sys
import pandas as pd
import pygame
import numpy as np
from PIL import Image
import re
from concurrent.futures import ThreadPoolExecutor
from ocatari.core import OCAtari
import time
import logging

# Initialize pygame
pygame.init()
pygame.display.init()

# Define output directories and files
output_dir = "pong1"
images_dir = os.path.join(output_dir, "images_v2")
annotations_file = os.path.join(output_dir, "atari_rgb_array2.csv")

# Create output directories if they don't exist
os.makedirs(images_dir, exist_ok=True)

game_name = "PongNoFrameskip-v4"
num_frames = 1000

# Initialize logging
logging.basicConfig(level=logging.DEBUG, filename='pong_debug.log', filemode='w')

logging.info("Starting the script")

try:
    env = OCAtari(game_name, mode="both", render_mode="rgb_array", hud=True)
    logging.info("OCAtari environment initialized successfully")
except Exception as e:
    logging.error(f'Error initializing OCAtari environment: {e}')
    sys.exit()

obs, info = env.reset()
data = []
previous_positions = {}

def extract_data(visual_desc):
    positions = {}
    sizes = {}
    bounding_boxes = {}
    object_patterns = re.findall(r"(\w+) at \((\d+), (\d+)\), \((\d+), (\d+)\)", visual_desc)

    for obj in object_patterns:
        label = obj[0]
        if "Score" in label:  # Skip entries containing 'Score'
            continue
        position = (int(obj[1]), int(obj[2]))
        size = (int(obj[3]), int(obj[4]))
        bounding_box = (position[0], position[1], size[0], size[1])
        positions[label] = position
        sizes[label] = size
        bounding_boxes[label] = bounding_box

    return positions, sizes, bounding_boxes

def get_color_from_region(image, position, size):
    if position is None or size is None:
        return None
    x, y = position
    w, h = size
    region = np.array(image)[y:y+h, x:x+w]
    if region.size == 0:
        return None
    color = np.mean(region, axis=(0, 1))
    return tuple(color.astype(int))

def calculate_distance(pos1, pos2):
    if pos1 is None or pos2 is None:
        return None
    x1, y1 = pos1
    x2, y2 = pos2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def calculate_relationships(positions, sizes):
    relationships = {}
    objects = list(positions.keys())
   
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i >= j:
                continue
            pos1 = positions[obj1]
            pos2 = positions[obj2]
            size1 = sizes[obj1]
            size2 = sizes[obj2]
           
            if pos1 and pos2 and size1 and size2:
                x1, y1 = pos1
                x2, y2 = pos2
                w1, h1 = size1
                w2, h2 = size2
               
                if y1 + h1 <= y2:
                    relationships[f"{obj1}_{obj2}"] = "above"
                elif y2 + h2 <= y1:
                    relationships[f"{obj1}_{obj2}"] = "below"
                elif x1 + w1 <= x2:
                    relationships[f"{obj1}_{obj2}"] = "left of"
                elif x2 + w2 <= x1:
                    relationships[f"{obj1}_{obj2}"] = "right of"
                else:
                    relationships[f"{obj1}_{obj2}"] = "overlapping"

    return relationships

def save_frame_data(frame_idx, obs, reward, terminated, action):
    global previous_positions
    try:
        image = Image.fromarray(obs)
        image_path = os.path.join(images_dir, f"{frame_idx}.png")
        image.save(image_path)
        logging.debug(f"Saved image {image_path}")

        positions, sizes, bounding_boxes = extract_data(str(env.objects))
        logging.debug(f"Frame {frame_idx}: positions - {positions}, sizes - {sizes}, bounding_boxes - {bounding_boxes}")
        colors = {key: get_color_from_region(obs, positions.get(key), sizes.get(key)) for key in positions}

        distances = {}
        objects = list(positions.keys())
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue
                distances[f"{obj1}_{obj2}"] = calculate_distance(positions.get(obj1), positions.get(obj2))

        relationships = calculate_relationships(positions, sizes)

        frame_data = {
            'frame_id': frame_idx,
            'positions': positions,
            'sizes': sizes,
            'bounding_boxes': bounding_boxes,
            'colors': colors,
            'distances': distances,
            'relationships': relationships
        }

        previous_positions.update(positions)
        return frame_data
    except Exception as e:
        logging.error(f"Error saving frame data: {e}")
        return None

def main():
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for frame_idx in range(num_frames):
            pygame.event.pump()
            logging.info(f"Processing frame {frame_idx}")

            positions, sizes, bounding_boxes = extract_data(str(env.objects))
            logging.info(f"Frame {frame_idx}: {positions}")

            action = env.action_space.sample()  # Random action for the game

            obs, reward, terminated, truncated, info = env.step(action)
            futures.append(executor.submit(save_frame_data, frame_idx, obs, reward, terminated, action))

            if terminated or truncated:
                obs, info = env.reset()

            pygame.time.wait(50)

        for future in futures:
            result = future.result()
            if result is not None:
                data.append(result)

    df = pd.DataFrame(data)
    try:
        df.to_csv(annotations_file, index=False)
        logging.info(f"Saved annotations to {annotations_file}")
    except Exception as e:
        logging.error(f"Error saving CSV file: {e}")

    env.close()
    pygame.quit()
    logging.info(f"Saved {num_frames} frames and annotations to {annotations_file}")

if __name__ == "__main__":
    main()
