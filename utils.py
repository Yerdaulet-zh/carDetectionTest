import tqdm, torch
import xml.etree.ElementTree as ET


def parse(root):
    name, pose, truncated, difficult, occluded, xmin, xmax, ymin, ymax = [], [], [], [], [], [], [], [], []


    for obj in root.findall("object"):
        name.append(obj.find('name').text)
        pose.append(obj.find("pose").text)
        truncated.append(obj.find("truncated").text)
        difficult.append(obj.find("difficult").text)
        occluded.append(obj.find("occluded").text)
        coors = obj.find("bndbox")
        xmin.append(int(coors.find("xmin").text))
        xmax.append(int(coors.find("xmax").text))
        ymin.append(int(coors.find("ymin").text))
        ymax.append(int(coors.find("ymax").text))
    return name, pose, truncated, difficult, occluded, xmin, xmax, ymin, ymax




def get_metadata(folder, annotation_names):
    validation = []
    # names = []
    for annotation_name in tqdm.tqdm(annotation_names): 
        file = []
        tree = ET.parse(f"dataset/{folder}/" + annotation_name)
        root = tree.getroot()
        name, pose, truncated, difficult, occluded, xmin, xmax, ymin, ymax = parse(root)    

        for i in range(len(name)):
            unit = []
            if name[i][:5] == 'truck':
                nam = 'truck'
            elif name[i] == "big bus":
                nam = "bus"
            elif name[i] == "small bus":
                nam = "bus"
            elif name[i] == "bus-l-":
                nam = "bus"
            elif name[i] == "bus-s-":
                nam = "bus"
            elif name[i] == "big truck":
                nam = "truck"
            elif name[i] == "mid truck":
                nam = "truck"
            elif name[i] == "small truck":
                nam = "truck"
            else: nam = name[i]
            unit.append(nam)
            unit.append(xmin[i])
            unit.append(ymin[i])
            unit.append(xmax[i])
            unit.append(ymax[i])
            # names.append(nam)
#             unit.append(ymin[i])
#             unit.append(xmin[i])
#             unit.append(ymax[i])
#             unit.append(xmax[i])            
            file.append(unit)
        validation.append(file)
    return validation



def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = (boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2) * 2
        box1_y1 = (boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2) * 2
        box1_x2 = boxes_preds[..., 0:1]# + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2]# + boxes_preds[..., 3:4] / 2
        box2_x1 = (boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2) * 2
        box2_y1 = (boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2) * 2
        box2_x2 = boxes_labels[..., 0:1]#  + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2]#  + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)



def corner_to_center(bbox):
    xmin, ymin, xmax, ymax = bbox
    xmin = (xmax + xmin) / 2
    ymin = (ymax + ymin) / 2
    return xmin, ymin, xmax, ymax


#@save https://d2l.ai/chapter_computer-vision/bounding-box.html
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes# boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    # boxes = torch.stack((cx, cy, w, h), axis=-1)
    return cx, cy, w, h # boxes


def center_to_corner(bbox):
    xmin, ymin, xmax, ymax = bbox
    xmin = (xmin - xmax // 2) * 2
    ymin = (ymin - ymax // 2) * 2
    return xmin, ymin, xmax, ymax


def bbox2yolo(bbox):
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return x, y, w, h


def yolo2bbox(bbox):
    x, y, w, h = bbox
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2

