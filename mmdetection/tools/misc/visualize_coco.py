# -*- coding: utf-8 -*-
from pycocotools.coco import COCO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from tqdm import tqdm

colors = [
    [], 
    [53, 119, 181],
    [245, 128, 6],
    [67, 159, 36],
    [204, 43, 41],
    [145, 104, 190],
    [135, 86, 75],
    [219, 120, 195],
    [127, 127, 127]
    # [204, 43, 41]
]
colors = [[i / 255 for i in c] for c in colors]
def showBBox(coco, anns, label_box=True, is_filling=True):
    """
    show bounding box of annotations or predictions
    anns: loadAnns() annotations or predictions subject to coco results format
    label_box: show background of category labels or not
    """
    if len(anns) == 0:
        return 0
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    image2color = dict()
    for _i, cat in enumerate(coco.getCatIds()):
        # image2color[cat] = (np.random.random((1, 3)) * 0.9 + 0.1).tolist()[0]
        # image2color[cat] = [i / 255.  for i in colors[_i]]
        image2color[cat] = colors[_i + 1]
    for ann in anns:
        c = image2color[ann['category_id']]
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)
        if label_box:
            label_bbox = dict(facecolor=c)
        else:
            label_bbox = None
        if 'score' in ann:
            ax.text(bbox_x, bbox_y, '%s: %.2f' % (coco.loadCats(ann['category_id'])[0]['name'], ann['score']),
                    color='white', bbox=label_bbox)
        else:
            ax.text(bbox_x, bbox_y, '%s' % (coco.loadCats(ann['category_id'])[0]['name']), color='white',
                    bbox=label_bbox)
    if is_filling:
        # option for filling bounding box
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


# only_bbox 为True表示仅仅可视化bbox，其余label不显示
# show_all 表示所有类别都显示，否则category_name来确定显示类别
def show_coco(data_root, ann_file, img_prefix, only_bbox=False, show_all=True, category_name='bicycle'):
    example_coco = COCO(ann_file)
    print('图片总数：{}'.format(len(example_coco.getImgIds())))
    categories = example_coco.loadCats(example_coco.getCatIds())
    category_names = [category['name'] for category in categories]
    print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

    if show_all:
        category_ids = []
    else:
        category_ids = example_coco.getCatIds(category_name)
    image_ids = example_coco.getImgIds(catIds=category_ids)

    for i in tqdm(range(len(image_ids))):
        # if str(image_ids[i]) != '3653': 
        #     continue
        # else:
        #     print(image_ids[i])
        plt.figure()
        image_data = example_coco.loadImgs(image_ids[i])[0]
        path = os.path.join(data_root, img_prefix, image_data['file_name'])
        image = cv2.imread(path)
        image = cv2.cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
        annotations = example_coco.loadAnns(annotation_ids)
        if only_bbox:
            showBBox(example_coco, annotations)
        else:
            # ###########################
            showBBox(example_coco, annotations, label_box=True, is_filling=False)
            example_coco.showAnns(annotations, color_dict=colors)
        plt.axis('off')
        plt.savefig('/home/ubuntu/code/LowLight/mmdetexp/DatasetVis/' + os.path.split(path)[1][:-4] + '.jpg', dpi=121, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None, 
                    transparent=False, bbox_inches='tight', pad_inches=0.0,
                    frameon=None, metadata=None)
        # exit()


if __name__ == '__main__':
    # 和cfg里面设置一样 coco
    
    data_root = '/home/ubuntu/2TB/dataset/LOD/'
    # ann_file = '/home/ubuntu/2TB/dataset/LOD/lis_coco_png_traintest.json'
    ann_file = '/home/ubuntu/2TB/dataset/LOD/coco.json'
    img_prefix = '/home/ubuntu/2TB/dataset/LOD/RGB_Normal/JPEGImages'
    show_coco(data_root, ann_file, img_prefix)
    exit()
    data_root = '/home/ubuntu/dataset/coco/'
    ann_file = data_root + 'annotations/instances_val2017.json'
    img_prefix = data_root + 'images/val2017/'
    show_coco(data_root, ann_file, img_prefix)
    exit()

    # voc转化为coco后显示
    data_root = '/home/ubuntu/dataset/VOCdevkit/'
    ann_file = data_root + 'annotations/voc0712_trainval.json'
    img_prefix = data_root
    show_coco(data_root, ann_file, img_prefix)