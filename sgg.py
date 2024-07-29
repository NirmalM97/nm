import csv
import json

def parse_positions(positions_str):
    return eval(positions_str)

def parse_bounding_boxes(bboxes_str):
    return eval(bboxes_str)

def parse_relations(relations_str):
    return eval(relations_str)

def convert_csv_to_coco(csv_file, output_json_file):
    images = []
    annotations = []
    categories = set(["__background__"])
    category_id_map = {"__background__": 0}
    annotation_id = 1
    relationships = []
    relationship_id_map = {"__background__": 0}
    relationship_id = 1

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_id = int(row['frame_id'])
            file_name = f"frame_{image_id}.png"
            
            positions = parse_positions(row['positions'])
            bounding_boxes = parse_bounding_boxes(row['bounding_boxes'])
            relations = parse_relations(row.get('relationships', '{}'))
            
            images.append({
                "id": image_id,
                "file_name": file_name,
                "height": 210,
                "width": 160
            })
            
            object_index_map = {}
            for idx, (category, bbox) in enumerate(bounding_boxes.items()):
                if category not in category_id_map:
                    category_id_map[category] = len(category_id_map)
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id_map[category],
                    "bbox": bbox,
                    "iscrowd": 0
                })
                object_index_map[category] = idx
                annotation_id += 1
                categories.add(category)

            for rel, relationship in relations.items():
                subject_category, object_category = rel.split('_')
                subject_index = object_index_map.get(subject_category)
                object_index = object_index_map.get(object_category)
                if subject_index is not None and object_index is not None:
                    if relationship not in relationship_id_map:
                        relationship_id_map[relationship] = len(relationship_id_map)
                    relationships.append({
                        "id": relationship_id,
                        "image_id": image_id,
                        "subject_index": subject_index,
                        "object_index": object_index,
                        "predicate": relationship_id_map[relationship]
                    })
                    relationship_id += 1
                else:
                    print(f"Warning: Invalid relationship {rel} in image {image_id}. Subject or object not found.")

    categories_list = [{"id": category_id_map[cat], "name": cat} for cat in categories]
    relationships_list = [{"id": relationship_id_map[rel], "name": rel} for rel in relationship_id_map]

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories_list,
        "relationships": relationships,
        "relation_categories": relationships_list
    }

    with open(output_json_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

# Example usage
convert_csv_to_coco(
    csv_file='/work/rleap1/nirmal.aheshwari/Pong/pong.csv',
    output_json_file='/work/rleap1/nirmal.aheshwari/Pong/annotations1.json'
)





