import sys
import time
import datetime
import torch
import argparse
import logging
import json
import os
from torch.utils.data import DataLoader
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import make_optimizer, make_lr_scheduler
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.comm import reduce_dict as reduce_loss_dict
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.bounding_box import BoxList
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from PIL import Image

def get_rank():
    return 0

def validate_tensor(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")

def verify_model_config(cfg):
    logging.info("Verifying model configuration...")
    backbone_out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    roi_box_conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    logging.info(f"Backbone output channels: {backbone_out_channels}")
    logging.info(f"ROI box head convolution dimensions: {roi_box_conv_head_dim}")
    logging.info(f"Using FPN: {use_fpn}")

    if use_fpn:
        logging.info("FPN is used, ensuring configurations match.")

def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, targets

def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0])
    ])

def make_data_loader(cfg, mode='train', is_distributed=False, start_iter=0, collate_fn=None):
    def transform(image, target):
        image = T.ToTensor()(image)
        image = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)(image)
        return image, target

    if mode == 'train':
        annotation_file = '/work/rleap1/nirmal.aheshwari/Pong/annotations1.json'
        img_dir = '/work/rleap1/nirmal.aheshwari/Pong/images_v2'
        dataset = CustomDataset(annotation_file, img_dir, transforms=transform)
        data_loader = DataLoader(dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH // 2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    else:
        pass
    return data_loader

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, img_dir, transforms=None):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transforms = transforms

        #validate_dataset(self.annotations)

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        image_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        annos = [anno for anno in self.annotations['annotations'] if anno['image_id'] == image_id]
        boxes = [anno['bbox'] for anno in annos]
        labels = [anno['category_id'] for anno in annos]

        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, (img_info['width'], img_info['height']), mode="xywh").convert("xyxy")
        target.add_field("labels", torch.as_tensor(labels))

        relations = self._get_relations(image_id, len(annos))
        target.add_field("relation", relations)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def _get_relations(self, image_id, num_boxes):
        relations = torch.zeros((num_boxes, num_boxes), dtype=torch.int64)
        for rel in self.annotations['relationships']:
            if rel['image_id'] == image_id:
                subject_index = rel['subject_index']
                object_index = rel['object_index']

            # Detailed logging for debugging
               # print(f"Processing relationship: {rel}")
               # print(f"Subject Index: {subject_index}, Object Index: {object_index}, Number of boxes: {num_boxes}")

            # Validation check to ensure subject and object indices are within bounds
                if 0 <= subject_index < num_boxes and 0 <= object_index < num_boxes:
                    predicate = rel['predicate']
                    relations[subject_index, object_index] = predicate
                else:
                    print(f"Invalid subject_index: {subject_index}, object_index: {object_index}, num_boxes: {num_boxes}")
        return relations


'''def validate_dataset(dataset):
    category_ids = {category['id'] for category in dataset['categories']}
    image_ids = {image['id'] for image in dataset['images']}
    annotation_ids = {annotation['id'] for annotation in dataset['annotations']}
    predicate_ids = {rel_cat['id'] for rel_cat in dataset['relation_categories']}

    for annotation in dataset['annotations']:
        assert annotation['category_id'] in category_ids, f"Invalid category_id: {annotation['category_id']}"

    for relationship in dataset['relationships']:
        assert relationship['image_id'] in image_ids, f"Invalid image_id: {relationship['image_id']}"
      #  assert relationship['subject_id'] in annotation_ids, f"Invalid subject_id: {relationship['subject_id']}"
       # assert relationship['object_id'] in annotation_ids, f"Invalid object_id: {relationship['object_id']}"
        assert relationship['predicate'] in predicate_ids, f"Invalid predicate: {relationship['predicate']}"

    print("Dataset validation completed successfully.")
'''


def train(cfg, logger):
    model = build_detection_model(cfg)
    data_loader = make_data_loader(cfg, mode="train", collate_fn=collate_fn)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model, logger)
    scheduler = make_lr_scheduler(cfg, optimizer, logger)

    scaler = GradScaler()
    arguments = {"iteration": 0}

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk, logger)

    data_loader = make_data_loader(cfg, mode='train', is_distributed=False, start_iter=arguments["iteration"], collate_fn=collate_fn)

    meters = MetricLogger(delimiter="  ")
    max_iter = cfg.SOLVER.MAX_ITER
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    model.train()

    for iteration, (images, targets) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration += 1
        arguments["iteration"] = iteration

        try:
            images = images.to(device)
            targets = [target.to(device) for target in targets]
        except Exception as e:
            logger.error(f"Error in image transformation: {e}")
            raise

        validate_tensor(images, "images")
        for i, target in enumerate(targets):
            validate_tensor(target.bbox, f"target[{i}].bbox")

        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        validate_tensor(losses, "losses")

        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpointer.save(f"model_{iteration:07d}", **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=int(total_training_time)))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def main():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device Count:", torch.cuda.device_count())
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU found.")
    print("starting training script....")
    sys.stdout.flush()
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--config-file", default="/work/rleap1/nirmal.aheshwari/Scene-Graph-Benchmark.pytorch/configs/myconfig0.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())

    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, logger)

if __name__ == "__main__":
    main()

