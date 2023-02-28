import math
import os
import pickle
from pathlib import Path
from multiprocessing import Pool
from copy import deepcopy
from sklearn import semi_supervised
import tqdm
from PIL import Image
import torch
import time
import copy

import numpy as np
import cv2
from pycocotools import mask as maskUtils

# from Transformer import Transformer

# from DB_cfg import *
# from my_utils import encode_bool_masks

# ...............................


# Model..............................

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode


# dataset_dir = './data/PA'

def load_mask_rcnn_model(model_name: str, part):
    cfg = get_cfg()

    # gpu index
    # cfg.MODEL.DEVICE = "cuda:1" if model_name != 'fisheye20220211-2' else "cuda:0"
    cfg.MODEL.DEVICE = "cuda:1"

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # set threshold for this model
    # print(f'{model_name}/model_final_{part}.pth')
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_{part}.pth'

    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_T2_{part}.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_FL_d{part}.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_FL_1st_f1{part}.pth'
    cfg.MODEL.WEIGHTS = f'{model_name}/model_final_FL_f{part}.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_newFL_e{part}.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_test_f2{part}.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_newFL_1st_f11{part}.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_f{part}.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_CE_f1{part}.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_CE_{part}.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_CnewFL_1{part}.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_FL_2{part}.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_0003239.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_02.pth'
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo


    # change the last nms
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

    # more topk per image
    cfg.TEST.DETECTIONS_PER_IMAGE = 10000

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8
    # return this predictor
    return DefaultPredictor(cfg)


# predictor = DefaultPredictor(cfg)
# outputs = predictor(P)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
# print(outputs["instances"].scores)
# print(outputs["instances"].pred_masks)

# copy results from gpu to cpu
#insts = outputs["instances"].to("cpu")


# ...............................


# def run_detection(model_name: str):
#     model = load_mask_rcnn_model(model_name)
#     print('model:', model, 'loaded......', flush=True)

#     for dbi, DB in enumerate(DB_LIST):
#         if dbi not in [7, 8, 11]:
#             print(f'{DB}: Maybe results are not good(?)', flush=True)
#         for vi in range(1, VideoNums[dbi] + 1):
#             for pim in Path('Combine_All').glob(f'image/{dbi}_{vi}_*.jpg'):
#                 print(f'\r{dbi} {vi}', end='')
#                 # Read image
#                 I = cv2.imread(str(pim))
#                 # Build Transform
#                 # Transform image
#                 T_I_list = [I]
#                 # Detect objects on transformed images with first hard NMS
#                 dets_list = [model(T_I) for T_I in T_I_list]
#                 # detectron2 format
#                 # copy results from gpu to cpu
#                 dets_list = [dets["instances"].to("cpu") for dets in dets_list]
#                 # only human
#                 dets_list = [dets[dets.pred_classes == 0] for dets in dets_list]
#                 # to np array, drop classes
#                 dets_list = [{'scores': dets.scores.numpy(), 'masks': dets.pred_masks.numpy()} for dets in dets_list]
#                 # encode masks
#                 dets_list = [{'scores': dets['scores'], 'masks': encode_bool_masks(dets['masks'])} for dets in dets_list]
#                 result_name = 'results' + (model_name[-1] if model_name[-2] == '-' else '')
#                 Path(f'result/pretrained/{result_name}/').mkdir(parents=True, exist_ok=True)
#                 with Path(f'result/pretrained/{result_name}/{pim.name[:-4]}.pkl').open('wb') as f:
#                     pickle.dump(dets_list[0], f)


# def load_fisheye_val():
    # return load_fisheye(dataset_dir, val_list[part])

def preds2rgb(preds_image_e):
    preds_image_rgb = np.zeros((preds_image_e.shape[0], preds_image_e.shape[1], 3), dtype=np.uint8)
    # preds_image_rgb[preds_image_e==0,:] = [0, 0, 0]
    # preds_image_rgb[preds_image_e==1,:] = [34, 139, 34]
    # preds_image_rgb[preds_image_e==2,:] = [255, 255, 0]
    # preds_image_rgb[preds_image_e==3,:] = [255, 0, 0]
    # preds_image_rgb[preds_image_e==4,:] = [255, 228, 181]
    # preds_image_rgb[preds_image_e==5,:] = [128, 128, 128]
    # preds_image_rgb[preds_image_e==6,:] = [255, 255, 255]
    # preds_image_rgb[preds_image_e==0,:] = [0, 0, 0] # 背景
    preds_image_rgb[preds_image_e==7,:] = [255, 255, 0] # 背景
    preds_image_rgb[preds_image_e==1,:] = [34, 139, 34] # 法郎值
    # preds_image_rgb[preds_image_e==2,:] = [0, 255, 255] # 牙本質
    preds_image_rgb[preds_image_e==3,:] = [0, 0, 255] # 牙隨
    preds_image_rgb[preds_image_e==4,:] = [181, 228, 255] # 人工
    preds_image_rgb[preds_image_e==5,:] = [128, 128, 128] # 齒曹骨
    preds_image_rgb[preds_image_e==6,:] = [255, 0, 0] # 蛀牙

    
    return preds_image_rgb

def eliminate_noise(preds_image_t):
    carries_image = np.zeros((preds_image_t.shape[0], preds_image_t.shape[1]), dtype=int)
    carries_image[preds_image_t == 6] = 1
    # print(np.bincount(carries_image.flatten()))
    Morphologic_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # dilation = cv2.dilate(preds_image_t.astype('uint8'), Morphologic_kernel)
    # erosion = cv2.erode(dilation.astype('uint8'), Morphologic_kernel)
    
    
    erosion = cv2.erode(carries_image.astype('uint8'), Morphologic_kernel)
    dilation = cv2.dilate(erosion.astype('uint8'), Morphologic_kernel)
    # print(np.bincount(dilation.flatten()))
    preds_image_t[preds_image_t == 6] = 7
    preds_image_t[dilation == 1] = 6
   

    # erosion = erosion.astype('int64')
    
    # preds_image_e = dilation
    # 
    return preds_image_t

def visualization(dets, name, ori):
    mask = dets['masks'].astype(int)
    bbox = dets['bbox']
    for i in range(mask.shape[0]):
        m = mask[i]
        b = bbox[i]
        rgb = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
        rgb[m == 1] = [255, 0, 0]
        cv2.rectangle(rgb, (b[0], b[1]), (b[2], b[3]), color=(0, 255, 0), thickness=5)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, np.hstack((ori, rgb)))
        cv2.waitKey()
    cv2.destroyAllWindows()


from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog, MetadataCatalog

def run_detection(model_name:str, data_path, cross_data_path, part):
    model = load_mask_rcnn_model(model_name, part)
    print('model:', model, 'loaded......', flush=True)
    # print(model)
    filedata = open(cross_data_path, 'rb')
    alldatalist = pickle.load(filedata)
    val_idx = alldatalist['val_idx'][part]
    print(val_idx)
    # _, cross_data = pickle.load(filedata)
    # val_idx = cross_data[part]

    # val_data = [os.listdir(data_path)[ix] for ix in val_idx]
    # DatasetCatalog.register("PA_val", load_fisheye_val)
    # building_metadata = MetadataCatalog.get('PA_val').set(thing_classes = ["tooth_ins"], stuff_classes=["empty", 'Alver', "tooth_sem"])
    building_metadata = MetadataCatalog.get('PA_val').set(thing_classes = ["dentin_th"], stuff_classes=["empty", "dentin_st", "enamel_st", 'pulp', 'artificial', 'Alveo', 'carries', 'Background'])
    # print(data_path)
    # for ix, val in enumerate(os.listdir(data_path)):
    total_time = 0
    for ix, val in enumerate(val_idx):
        img = cv2.imread(data_path+val)
        sem_gt = cv2.imread('./data/CutImage/sem_seg/T5/'+ os.path.splitext(val)[0] + '.png', cv2.IMREAD_GRAYSCALE)
        # sem_gt = cv2.imread('./data/PA/sem_seg/T5/'+ os.path.splitext(val)[0] + '.png', cv2.IMREAD_GRAYSCALE)
        print(val)
        # dets_list = [model(img)]
        start_time = time.clock()
        dets = model(img)
        total_time += (time.clock()-start_time)
        # print(torch.argmax(dets['sem_seg'], dim=0), '\n')
        # print(dets['instances'], '\n')
        # print(dets['panoptic_seg'], '\n')
        
        # print(dets['sem_seg'].shape)
        # print(dets['sem_seg'])
        panoptic_seg, segments_info = model(img)['panoptic_seg']
        # print(np.bincount(panoptic_seg.to('cpu').flatten()))
        # print(panoptic_seg)
        # print(segments_info)
        # try:
        #     mask = model(img)['pred_masks'].numpy()
        #     print('\n mask:', mask)
        # except:
        #     continue
        # print(mask)
        # dets_list = [dets["instances"].to("cpu") for dets in dets_list]

        # print("Before:\n")
        
        # print(segments_info)
        # print(len(segments_info))

        v1 = Visualizer(img[:, :, ::-1],
                        scale=1,
                        metadata=building_metadata,
           
                        instance_mode = ColorMode.IMAGE_BW,
        )
        v1 = v1.draw_panoptic_seg_predictions(panoptic_seg.to('cpu'), segments_info)

        delete_idx = []
        for idx, info in enumerate(segments_info):  
            if info['isthing'] == False:
                # if info['category_id'] == 1:
                panoptic_seg[panoptic_seg == info['id']] = 0
                delete_idx.append(idx)

        new_segments_info = [segments_info[idx] for idx in range(len(segments_info)) if idx not in delete_idx]
        

        # print("After:\n")
        # if val == '00000003_25.jpg':
            # print(new_segments_info)
        # print(len(new_segments_info))



        v2 = Visualizer(img[:, :, ::-1],
                        scale=1,
                        metadata=building_metadata,
                        instance_mode = ColorMode.IMAGE_BW,
        )

        v3 = Visualizer(img[:, :, ::-1],
                        scale=1,
                        metadata=building_metadata,
                        instance_mode = ColorMode.IMAGE_BW,
        )



        sem_rgb = preds2rgb(sem_gt)
        # sem_pred = torch.softmax(dets['sem_seg'], dim=0).to('cpu').numpy()
        sem_pred = dets['sem_seg'].to('cpu').numpy()
        # print(np.sum(sem_pred, axis=0))
        # sem_pred = np.sqrt(sem_pred)
        pred_map = np.argmax(sem_pred, axis=0)
        # pred_map[sem_pred[6, :, :] > 0.5] = 6
        print("before:" , np.bincount(pred_map.flatten()))
        pred_map = eliminate_noise(pred_map)
        print("After:", np.bincount(pred_map.flatten()))
        sem_pred_rgb = preds2rgb(pred_map)

        inst_pred = dets["instances"].to("cpu")
        # if val == '00000003_25.jpg':
            # print(inst_pred)
            # print(inst_pred.scores[0:len(new_segments_info)])
        # print(inst_pred.scores)


        obj = detectron2.structures.Instances(image_size=(img.shape[0], img.shape[1]))
        obj.set('pred_boxes', inst_pred.pred_boxes[0:len(new_segments_info)])
        obj.set('scores', inst_pred.scores[0:len(new_segments_info)])
        obj.set('pred_classes', inst_pred.pred_classes[0:len(new_segments_info)])
        obj.set('pred_masks', inst_pred.pred_masks[0:len(new_segments_info)])
        # if val == '00000003_25.jpg':
        #     print('obj:', obj)

        # inst_pred.num_instances = len(new_segments_info)
        # inst_pred.scores = inst_pred.scores[0:len(new_segments_info)]
        # inst_pred.scores = inst_pred.scores[0:len(new_segments_info)]
        # inst_pred.pred_masks = inst_pred.pred_masks.numpy()[:len(new_segments_info)]
        # inst_pred.pred_boxes = inst_pred.pred_boxes.numpy()[:len(new_segments_info)]

        # inst_dict = {'scores': inst_pred.scores.numpy(), 'masks': inst_pred.pred_masks.numpy(), 'bbox':inst_pred.pred_boxes.tensor.numpy()}
        inst_dict = {'scores': obj.scores.numpy(), 'masks': obj.pred_masks.numpy(), 'bbox':obj.pred_boxes.tensor.numpy()}
        # print(inst_dict)

        v2 = v2.draw_instance_predictions(obj.to("cpu"))
        v3 = v3.draw_sem_seg(torch.argmax(dets['sem_seg'], dim=0).to("cpu"))

        v4 = Visualizer(sem_pred_rgb,
                        scale=1,
                        metadata=building_metadata,
                        # instance_mode = ColorMode.IMAGE_BW,
        )
        # v4 = v4.draw_instance_predictions(dets["instances"].to("cpu"))
        v4 = v4.draw_panoptic_seg_predictions(panoptic_seg.to('cpu'), new_segments_info)
        # v4.get_image()
        
        # folder = 'PAPanoptic'
        folder = 'BWPanoptic'

        Path(f'result/pretrained/{folder}pred/T5/').mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f'result/pretrained/{folder}pred/T5/{val}', np.hstack((img, sem_rgb, v2.get_image(), v1.get_image(), sem_pred_rgb, v4.get_image())))
        # cv2.imwrite(f'result/pretrained/Panoptic220817pred/{val}', v1.get_image()[:, :, ::-1])
        # # plt.imshow(v.get_image[:, :, ::-1])
        # cv2.imshow('', np.hstack((v1.get_image
        # ()[:, :, ::-1], v2.get_image()[:, :, ::-1], v4.get_image()[:, :, ::-1])))
        # cv2.imshow('', np.hstack((sem_rgb, v1.get_image(), v2.get_image(), sem_pred_rgb, v4.get_image())))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        dets_list = {'panoptic_seg':panoptic_seg.to('cpu').numpy(), 'segments_info':segments_info, 'sem_seg':pred_map, 'inst_seg':inst_dict}
        # print(dets_list)
        
        # dets_list = [dets[dets.pred_classes == 0] for dets in dets_list]
        # # print(dets_list)
        
        # dets_list = [{'scores': dets.scores.numpy(), 'masks': dets.pred_masks.numpy(), 'bbox':dets.pred_boxes.tensor.numpy()} for dets in dets_list]
        # # print(dets_list[0]['bbox'])
        # # print(dets_list[0]['masks'].shape)
        # # visualization(dets_list[0], val, img)
        # result_name = 'results_' + folder
        # Path(f'result/pretrained/{result_name}/T5/').mkdir(parents=True, exist_ok=True)
        # with Path(f'result/pretrained/{result_name}/T5/{val[:-4]}.pkl').open('wb') as f:
        #     pickle.dump(dets_list, f)

    return total_time

def run_eval(gt_path, detect_path):
    for pkl in os.listdir(gt_path):
        with Path(os.path.join(gt_path, pkl)).open('wb') as f:
            X = pickle.load(f)


if __name__ == '__main__':
    # cross_data_path = './CV_weight/PA_cross_data_316.pkl'
    # data_path = './data/PA/image/'
    
    cross_data_path = './CV_weight/BW_panoptic_cross_data_214.pkl'
    data_path = './data/CutImage/image/'
    
    total_time = 0
    for i in range(0, 4):
        total_time += run_detection('./checkpoints/PAPanoptic/T5', data_path=data_path, cross_data_path=cross_data_path, part=i)
        # total_time += run_detection('./checkpoints/BWPanoptic/T5', data_path=data_path, cross_data_path=cross_data_path, part=i)

    print(f'Total_time = {total_time}')
    # print(f'Inference_time(s/img) = {total_time / 316}')
    # print(f'Inference_time(s/img) = {total_time / 216}')
    print(f'Inference_time(s/img) = {total_time / 214}')
    

    # img = cv2.imread('./data/image/04561074_20190502_27_up.jpg')
    # # print(img)
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.WEIGHTS = './BWMaskRcnn220615/model_final_1.pth'
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    # predictor = DefaultPredictor(cfg)
    # outputs = predictor(img)
    # print(outputs)
    # outputs = model(img)
    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)
    # print(outputs["instances"].scores)
    # print(outputs["instances"].pred_masks)   
    # run_detection('')
    # pool = Pool()
    # aa = pool.map_async(run_detection, ['pretrained'])  # gpu 1
    # pool.map_async(run_detection, ['fisheye20220211-2'])  # gpu 0
    # aa.get()
    # pool.map_async(run_detection, ['fisheye20220211-3'])  # gpu 1
    # pool.close()
    # pool.join()
