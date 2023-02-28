import numpy as np
import pickle
import LabelData
from panopticapi.utils import rgb2id, id2rgb
import os
import copy
import cv2
from PIL import Image

img_path ='data/CutImage/image/'
instance_path = 'data/CutImage/instance_label/'
semantic_path = 'data/CutImage/semantic_label/'
output_panoptic = 'data/CutImage/panoptic_label/T5/'

def get_anno(label):
    anno = np.zeros((label['MapPixel2RegionId'].shape[0], label['MapPixel2RegionId'].shape[1], len(label['ListRegion'])), int)
    for ix in range(len(label['ListRegion'])):
        t_listid = label['ListRegion'][ix].listRegionId

        for id in t_listid:
            anno[label['MapPixel2RegionId'] == id, ix] = 1

    return anno

def get_panoptic_anno(ins, seg):
    ins_label = get_anno(ins)
    # seg_label = get_anno(seg)

    # pan_map = np.zeros((seg['MapPixel2RegionId'].shape[0], seg['MapPixel2RegionId'].shape[1]), int)
    pan_map = copy.deepcopy(seg)
    # pan_map = copy.deepcopy(seg['MapPixel2RegionId'])
    # print(ins_label.shape[2])
    for ix in range(0, ins_label.shape[2]):
        seg_map = copy.deepcopy(seg)

        fullteeth = ins_label[..., ix]
        # seg_map[fullteeth == 0] = 0

        # for cate in range(1, seg_label.shape[2]):
            # seg_map[pan_map == cate] = ix*10 + cate
        # print(np.bincount(seg_map.flatten()))
        # print(np.bincount(seg_map.flatten()).shape)
        # print()
        # pan_map[seg_map != 0] = seg_map[seg_map != 0]
        pan_map[fullteeth != 0] = ix*10 + 1
        # pan_map[seg_map == 1] = ix*10 + 1
        # pan_map[seg_map == 2] = ix*10 + 2
        # print(np.bincount(pan_map.flatten()))
        # print(np.bincount(pan_map.flatten()).shape)
        # print()
    # rgb = id2rgb(pan_map)
    # rgb = rgb*20
    # print(np.bincount(pan_map.flatten()))
    # print(np.bincount(rgb.flatten()))
    # print(np.bincount(pan_map.flatten()).shape)
    # exit()
    # cv2.imshow('', rgb)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return pan_map

'''
220817 anno: 1.fullteeth, 2.alveolar bone, 3.background
'''


for img in os.listdir(img_path):
    filename, _ = os.path.splitext(img)
    print(filename)
    
    instance_file = open(instance_path+filename+'.pkl', 'rb')
    instance_label = pickle.load(instance_file)
    instance_map = copy.deepcopy(instance_label['MapPixel2RegionId'])
    instance_map[instance_map != 0] = 1

    semantic_file = open(semantic_path+filename+'.pkl', 'rb')
    semantic_label = pickle.load(semantic_file)

    # ========== 0.Ingore, 1.Fullteeth, 2.Alveolar bone, 3.background ==========
    # seg_label = copy.deepcopy(semantic_label['MapPixel2RegionId'])
    # seg_label[seg_label != 5] = 3
    # seg_label[seg_label == 5] = 2
    # seg_label[instance_map != 0] = 1

    # ========== 0.Ingore, 1.Fullteeth, 2. Enamel, 3.Alveolar bone, 4.background ==========
    # seg_temp_label = copy.deepcopy(semantic_label['MapPixel2RegionId'])
    # seg_label = np.zeros((seg_temp_label.shape[0], seg_temp_label.shape[1]), dtype=int)
    # seg_label[seg_temp_label == 5] = 3
    # seg_label[instance_map != 0] = 1
    # seg_label[seg_temp_label == 1] = 2
    # seg_label[seg_label == 0] = 4

    # ========== 0.Ingore, 1.Fullteeth, 2. Enamel, 3.Pulp, 4.Artificial, 5.Alveolar bone, 6.carries, 7.background ==========
    seg_label = copy.deepcopy(semantic_label['MapPixel2RegionId'])
    seg_label[seg_label == 0] = 7

    pil_img_gray = Image.fromarray(np.uint8(seg_label))
    pil_img_gray.save('data/CutImage/sem_seg/T5/'+filename+'.png')
    # np.save(semantic_path+filename+'_label', seg_label)
    # print(np.bincount(seg_label.flatten()))
    # cv2.imwrite(semantic_path+filename+'_label.jpg', seg_label.astype(np.uint8))
    a = cv2.imread('data/CutImage/sem_seg/T5/'+filename+'.png', cv2.IMREAD_GRAYSCALE)
    print(np.bincount(a.flatten()))

    instance_file.close()
    semantic_file.close()
    
    pan_anno = get_panoptic_anno(instance_label, seg_label)
    num = np.bincount(pan_anno.flatten())
    listRegion = []
    new_dict = {}
    for i in range(1, num.shape[0]):
        if num[i] != 0:
            listRegion.append(LabelData.ROIData(i, [i]))
    
    new_dict['MapPixel2RegionId'] = pan_anno
    new_dict['ListRegion'] = listRegion
    save_file = open(os.path.join(output_panoptic, filename+'.pkl'), 'wb')
    pickle.dump(new_dict, save_file)
    save_file.close()
    print(np.bincount(pan_anno.flatten()))
    # np.save(output_panoptic+filename, pan_anno)
    # cv2.imwrite(output_panoptic+filename+'.jpg', id2rgb(pan_anno))
