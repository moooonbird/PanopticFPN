import math
import os
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import cv2
import copy
from pycocotools import mask as maskUtils
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# from Transformer import Transformer

# from DB_cfg import *
# from my_utils import encode_bool_masks
from mAP_example import match_masks_stuff, compute_eval2, compute_eval2_thing, compute_eval2_stuff, match_masks_thing
import LabelData

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



#predictor = DefaultPredictor(cfg)
#outputs = predictor(P)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
# print(outputs["instances"].scores)
# print(outputs["instances"].pred_masks)

# copy results from gpu to cpu
#insts = outputs["instances"].to("cpu")


# ...............................

def eval_classification_now(nclass, cf):
    print(np.sum(cf * np.eye(nclass), axis=1))
    print(np.maximum(np.sum(cf, axis=1), 1))
    print(np.maximum(np.sum(cf, axis=0), 1))
    sensitivity = np.sum(cf * np.eye(nclass), axis=1) / np.maximum(np.sum(cf, axis=1), 1)
    print(sensitivity)
    recall = sensitivity
    precision = np.sum(cf * np.eye(nclass), axis=0) / np.maximum(np.sum(cf, axis=0), 1)
    F1 = (recall * precision * 2) / (recall + precision + 1e-10)

    # AC = compute_class_accuracy(nclass, cf, 0)
    
    return [precision, sensitivity, F1]

def decode_bool_masks(masks):
    if type(masks) == list:
        masks = maskUtils.decode(masks)  # (H, W, N), column major
        masks = np.ascontiguousarray(np.transpose(masks, [2, 0, 1]))  # (N, H, W), row-major
    return masks.astype(np.bool)  # bool

def retrieve_carries(ori_img, fullteeth=None, carries=0, carries_threshold=0.003):
    image = copy.deepcopy(ori_img.astype('uint8'))
    if carries == 0:
        image[image != 6] = 0
    
    image[fullteeth == 0] = 0
    num, label, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    # print(stats)
    area = image.shape[0] * image.shape[1]
    # print('\n before: ', num)
    # print(np.bincount(label.flatten()))
    for idx in range(1, num):
        carriesarea = stats[idx][-1]
        ratio = carriesarea / area
        if ratio < carries_threshold:
            label[label == idx] = 0
            num -= 1
            continue
    
    le = LabelEncoder()
    le.fit(label.flatten())
    temp = le.transform(label.flatten())
    # print('\nSort: ', np.bincount(temp))
    temp = np.reshape(temp, (ori_img.shape[0], ori_img.shape[1]))
    
    return num, temp

def labeling(num_GT, GT, num_Pred, Pred):
    iou_list = []
    Pred_list = []
    # print(np.bincount(GT.flatten()))
    label_list = []
    check_pred = np.zeros(num_Pred, dtype=int)
    for idx_GT in range(1, num_GT):
        GT_label = np.zeros((GT.shape[0], GT.shape[1]), dtype=int)
        GT_label[GT == idx_GT] = 1
        label_dict = {}
        label_dict['Pred'] = []

        for idx_Pred in range(1, num_Pred):
            Pred_label = np.zeros((GT.shape[0], GT.shape[1]), dtype=int)
            Pred_label[Pred == idx_Pred] = 1
            
            p_map = GT_label + Pred_label
            universe = np.bincount(p_map.flatten())
            # print("NP_WHERE:", np.argwhere(p_map == 2).shape)
            # print('Universe:', universe)
            if universe.shape[0] < 3:
                intersect = 0
                iou = 0
            else:
                intersect = universe[2]
                union = universe[1] + universe[2]
                iou = float(intersect / union)
 
            if iou > 0 and check_pred[idx_Pred] == 0:    
                label_dict['Pred'].append(idx_Pred)
                check_pred[idx_Pred] = 1


        label_list.append(label_dict)

    return label_list


def run_eval_pkl(panoptic_gt_path, instance_gt_path, sem_gt_path, detect_path):
    file = open('./CV_weight/BW_panoptic_cross_data.pkl', 'rb')
    filedata = pickle.load(file)
    val_idx = filedata['val_idx'][0]
    iou_threshold = np.arange(0.5, 1.00, 0.05)
    # iou_threshold = [0.5]
    total_ap = 0
    for threshold in iou_threshold:
        print(f'threshold={threshold}')
        ## ===== For Instaance Segmentation Metrics =====
        N_GT, TP_scores, FP_scores = 0, [], []

        ## ===== For Panoptic Segmentation Metrics =====
        total_pred = 0
        total_gt = 0
        TP_carries = 0

        ## ===== For Panoptic Segmentation Metrics =====
        N_GT_thing, N_FN_thing, TP_scores_thing, FP_scores_thing, TP_iou_thing = 0, 0, [], [], []
        N_GT_stuff, N_FN_stuff, N_TP_stuff, N_FP_stuff, TP_iou_stuff = 0, 0, 0, 0, []
        
        N_GT_all, N_FN_all, N_TP_all, N_FP_all, TP_iou_all = 0, 0, 0, 0, []

        N_GT_all, N_FN_A, N_TP_A, N_FP_A, TP_iou_A = 0, 0, 0, 0, []
        confusion_matrix = np.zeros((8, 8), dtype=int)
        # for i, pkl in enumerate(gt_path):
        # for i, pkl in enumerate(val_idx):
        
        for i, pkl in enumerate(os.listdir(panoptic_gt_path)):
            print(f'\r{i}, {pkl}', end='')
            with open(os.path.join(panoptic_gt_path, os.path.splitext(pkl)[0]+'.pkl'), 'rb') as f:
                GT = pickle.load(f)
                # print(GT)

            with open(os.path.join(detect_path, os.path.splitext(pkl)[0]+'.pkl'), 'rb') as f:
                DT = pickle.load(f)
                
            # print(GT['ListRegion'][0].listRegionId)
            ##### ========== Instance Segmentation Metrics  ========== ######
            with open(os.path.join(instance_gt_path, os.path.splitext(pkl)[0]+'.pkl'), 'rb') as f:
                GT_instance = pickle.load(f)
            gt_masks = np.zeros((len(GT_instance['ListRegion']), GT_instance["MapPixel2RegionId"].shape[0], GT_instance["MapPixel2RegionId"].shape[1]), dtype=bool)
            for i in range(len(GT_instance['ListRegion'])):
                t_listRegionId = GT_instance['ListRegion'][i].listRegionId
                for j in range(len(t_listRegionId)):
                    gt_masks[i, GT_instance['MapPixel2RegionId'] == t_listRegionId[j]] = True


            dt_inst = DT['inst_seg']
            scores = dt_inst['scores']
            dt_masks = dt_inst['masks']

            # panoptic_seg = DT['panoptic_seg']
            # segments_info = DT['segments_info']


            # dt_masks, scores = [], []
            # for info in segments_info:
            #     if info['isthing'] is True:
            #         temp = np.zeros((panoptic_seg.shape[0], panoptic_seg.shape[1]), dtype=bool)
            #         temp[panoptic_seg == info['id']] = True
            #         dt_masks.append(temp)
            #         scores.append(info['score'])
            
            

            # if len(dt_masks) == 0:
            #     dt_masks = np.zeros((1, GT["MapPixel2RegionId"].shape[0], GT["MapPixel2RegionId"].shape[1]), dtype=bool)
            #     scores.append(0)

            # dt_masks = np.array(dt_masks)
            # scores = np.array(scores)

            flagTP, gt_used, _ = match_masks_thing(dt_masks, gt_masks, threshold)
            # print('\n ====Instance Segmentation Metrics======')
            # print(flagTP)
            # print(gt_used)
            N_GT += len(gt_used)
            TP_scores.append(scores[flagTP])
            FP_scores.append(scores[np.logical_not(flagTP)])

            ##### ============= Panoptic Segmentation Metrics ========== ######
            # ========== Thing Part ==========    
            # gt_masks = np.zeros((len(GT['ListRegion'])-2, GT["MapPixel2RegionId"].shape[0], GT["MapPixel2RegionId"].shape[1]), dtype=bool)
            gt_masks = []
            for i in range(len(GT['ListRegion'])):
                temp = np.zeros((GT["MapPixel2RegionId"].shape[0], GT["MapPixel2RegionId"].shape[1]), dtype=bool)
                # check = np.zeros((GT["MapPixel2RegionId"].shape[0], GT["MapPixel2RegionId"].shape[1], 3), dtype=np.uint8)
                t_listRegionId = GT['ListRegion'][i].listRegionId
                if t_listRegionId[0] %10 != 1:
                    continue
        
                for j in range(len(t_listRegionId)):
                    # gt_masks[i, GT['MapPixel2RegionId'] == t_listRegionId[j]] = True
                    temp[GT['MapPixel2RegionId'] == t_listRegionId[j]] = True
                    # check[GT['MapPixel2RegionId'] == t_listRegionId[j], :] = [255, 0, 0]
                # cv2.imshow('', check)
                # cv2.waitKey()
                # cv2.destroyAllWindows
                gt_masks.append(temp)
            if len(gt_masks) == 0:
                gt_masks = np.zeros((1, GT["MapPixel2RegionId"].shape[0], GT["MapPixel2RegionId"].shape[1]), dtype=bool)

            gt_masks = np.array(gt_masks)
            # print(gt_masks.shape)

            panoptic_seg = DT['panoptic_seg']
            segments_info = DT['segments_info']


            dt_masks, scores = [], []
            for info in segments_info:
                if info['isthing'] is True:
                    temp = np.zeros((panoptic_seg.shape[0], panoptic_seg.shape[1]), dtype=bool)
                    temp[panoptic_seg == info['id']] = True
                    dt_masks.append(temp)
                    scores.append(info['score'])
            
            

            if len(dt_masks) == 0:
                dt_masks = np.zeros((1, GT["MapPixel2RegionId"].shape[0], GT["MapPixel2RegionId"].shape[1]), dtype=bool)
                scores.append(0)
            dt_masks = np.array(dt_masks)
            scores = np.array(scores)

            flagTP_thing, gt_used_thing, iou = match_masks_thing(dt_masks, gt_masks, 0.5)
            TP_iou_thing += iou
            # print()
            # print("--------Thing Part------")
            # print("Flag_TP_thing: ", flagTP_thing)
            # print("GT_used_thing: ", gt_used_thing)
            N_GT_thing += len(gt_used_thing)
            N_FN_thing += np.sum(np.array(gt_used_thing) == 0)

            # print(flagTP)
            TP_scores_thing.append(scores[flagTP_thing])
            FP_scores_thing.append(scores[np.logical_not(flagTP_thing)])

            # ========== Stuff Part ========== 
            ## stuff:{Alveolar_bone, Background}

            GT = cv2.imread(os.path.join(sem_gt_path, os.path.splitext(pkl)[0]+'.png'), cv2.IMREAD_GRAYSCALE)
            ori_img = cv2.imread(os.path.join('./data/CutImage/image', os.path.splitext(pkl)[0]+'.jpg'), cv2.IMREAD_GRAYSCALE)
            listregion = np.bincount(GT.flatten())
            # gt_masks = []
            gt_masks = np.zeros((8, GT.shape[0], GT.shape[1]), dtype=bool)
            for i in range(listregion.shape[0]):
                # if i % 10 == 1 or i % 10 == 0:
                if i % 10 != 5 and i % 10 != 7:
                    continue
                if listregion[i] == 0 :
                    continue

                gt_masks[i, GT == i] =True

                # temp = np.zeros((GT.shape[0], GT.shape[1]), dtype=bool)
                # temp[GT == i] = True   
            
                # gt_masks.append(temp)

            # if len(gt_masks) == 0:
            #     gt_masks = np.zeros((1, GT.shape[0], GT.shape[1]), dtype=bool)
            # gt_masks = np.array(gt_masks)

            # dt_masks = []
            sem_seg = DT['sem_seg']
            carries_seg = copy.deepcopy(sem_seg)
            dt_masks = np.zeros((8, sem_seg.shape[0], sem_seg.shape[1]), dtype=bool)
            listregion = np.bincount(sem_seg.flatten())
            for i in range(listregion.shape[0]):
                # if i == 0:
                if i != 5 and i != 7:
                    continue
                if listregion[i] == 0:
                    continue

                dt_masks[i, sem_seg == i] = True
                # temp = np.zeros((sem_seg.shape[0], sem_seg.shape[1]), dtype=bool)
                # temp[sem_seg == i] = True
                # dt_masks.append(temp)
            
            # carries_seg[ori_img < 25] = 0

            ##======================== Carries Evalutation====================
            '''
            num_pred, label_pred = retrieve_carries(carries_seg, carries_threshold=0)
            num_GT, label_GT = retrieve_carries(GT, carries_threshold=0)

            match_list = labeling(num_GT, label_GT, num_pred, label_pred)
            num_pred -= 1
            num_GT -= 1
            total_pred += num_pred
            total_gt += num_GT
            print()
            print("Num_pred:", num_pred, " Num_GT:", num_GT)

            '''
                #match:[idx_pred]
            '''
            print("match:", match_list)
            check = []
            for idx_GT, match in enumerate(match_list):
                Pred_list = match['Pred']                
                if len(Pred_list) != 0:
                    TP_carries += 1
                    for idx_pred in Pred_list:
                        print(idx_pred)
                        # if idx_pred not in check:
                        #     check.append(idx_pred)
                        # else:

                        carries_seg[label_pred == idx_pred] = 8

                    total_pred -= (len(Pred_list) - 1)
                    num_pred -= (len(Pred_list) - 1)
        
            print("Num_pred:", num_pred, " Num_GT:", num_GT)
            '''
            # if len(dt_masks) == 0:
            #     dt_masks = np.zeros((1, sem_seg.shape[0], sem_seg.shape[1]), dtype=bool)
            # dt_masks = np.array(dt_masks)

            flagTP_stuff, gt_used_stuff, flagFP_stuff, iou_stuff = match_masks_stuff(dt_masks, gt_masks)
            # print(flagTP_stuff, gt_used_stuff)
            # print("--------Stuff Part------")
            # print("FlagTP_stuff:", flagTP_stuff)
            # print("gt_used_stuff:", gt_used_stuff)
            # print()
            # print(np.sum(np.array(gt_used_stuff) == 0))
            N_GT_stuff += len(gt_used_stuff)
            for idx, (i, j) in enumerate(zip(flagTP_stuff, gt_used_stuff)):
                if i is False and j is True:
                    N_FN_stuff += 1
                if i is True and j is True:
                    
                    N_TP_stuff += 1
                    # TP_iou_stuff.append(iou_stuff[i])

            # N_FN_stuff += np.sum(np.array(gt_used_stuff) == 0)
            # N_TP_stuff += np.sum(np.array(gt_used_stuff) == 1)
            N_FP_stuff += np.sum(flagFP_stuff)

            TP_iou_stuff += iou_stuff

            for i in range(1, 8):
                for j in range(1, 8):
                    confusion_matrix[i][j] += np.sum(sem_seg[GT == i] == j)

            # print()
            # print("=======Only stuff part=======")
            # print("Flag_TP_stuff: ", flagTP_stuff)
            # print("GT_used_stuff: ", gt_used_stuff)
            # print("flagFP_stuff: ", flagFP_stuff)
            # print(N_TP_stuff, N_FN_stuff, N_FP_stuff)
            # print(N_FP_stuff)
            # print(flagTP)

            ### =========ALL Part=========
            ### Thing:{Fullteeth}, stuff:{except fullteeth}
            gt_masks = np.zeros((8, GT.shape[0], GT.shape[1]), dtype=bool)
            listregion = np.bincount(GT.flatten())
            for i in range(listregion.shape[0]):
                if i % 10 == 1 or i % 10 == 0:
                # if i % 10 != 5 and i % 10 != 7:
                    continue
                if listregion[i] == 0 :
                    continue

                gt_masks[i, GT == i] =True

            dt_masks = np.zeros((8, sem_seg.shape[0], sem_seg.shape[1]), dtype=bool)
            listregion = np.bincount(sem_seg.flatten())
            for i in range(listregion.shape[0]):
                if i == 0 or i == 1:
                # if i != 5 and i != 7:
                    continue
                if listregion[i] == 0:
                    continue

                dt_masks[i, sem_seg == i] = True

            flagTP_all, gt_used_stuff, flagFP_stuff, iou_stuff = match_masks_stuff(dt_masks, gt_masks)
            for idx, (i, j) in enumerate(zip(flagTP_all, gt_used_stuff)):
                if i is False and j is True:
                    N_FN_all += 1
                if i is True and j is True:
                    
                    N_TP_all += 1

            TP_iou_all += iou_stuff
            N_FP_all += np.sum(flagFP_stuff)
            # print()
            # print("======ALL part======")
            # print("Flag_TP_stuff: ", flagTP_all)
            # print("GT_used_stuff: ", gt_used_stuff)
            # print("flagFP_stuff: ", flagFP_stuff)
            # print(np.sum(flagFP_stuff))
            # print("TP_iou_stuff: ", TP_iou_stuff)
            # print(N_TP_all, N_FN_all, N_FP_all)

        
        # if TP_carries == 0:
        #     precision = 0
        #     recall = 0
        # else:
        #     precision = TP_carries / total_pred
        #     recall = TP_carries / total_gt

        # print("-------Carries Part:---------")
        # print("Num_gt_carries:", total_gt)
        # print("Num_pred_carries:", total_pred)
        # print("Num_TP_carries:", TP_carries)
        # print('Precision:', precision)
        # print('Recall:', recall, '\n')
        # print('F1_score:', 2*(precision*recall)/(precision+recall))


        [precision, sensitivity, F1] = eval_classification_now(8, confusion_matrix) 

        TP_scores_thing = np.concatenate(TP_scores_thing)
        FP_scores_thing = np.concatenate(FP_scores_thing)  
        
        PQ_stuff_B, SQ_stuff_B, RQ_stuff_B = compute_eval2_stuff(N_GT_stuff, N_FN_all, N_TP_all, N_FP_all, TP_iou_all, 'result/pretrained/Panoptic_stuff')
        PQ_stuff, SQ_stuff, RQ_stuff = compute_eval2_stuff(N_GT_stuff, N_FN_stuff, N_TP_stuff, N_FP_stuff, TP_iou_stuff, 'result/pretrained/Panoptic_stuff')


        N_TP_all += TP_scores_thing.shape[0]
        N_FP_all += FP_scores_thing.shape[0]
        N_FN_all += N_FN_thing
        TP_iou_all += TP_iou_thing

        N_TP_stuff += TP_scores_thing.shape[0]
        N_FP_stuff += FP_scores_thing.shape[0]
        N_FN_stuff += N_FN_thing
        TP_iou_stuff += TP_iou_thing

        # print('compute_eval')
        # # calculate precision, recall and VOC AP
        ap_thing, mrec_thing, mpre_thing, PQ_thing, SQ_thing, RQ_thing = compute_eval2_thing(N_GT_thing, N_FN_thing, TP_scores_thing, FP_scores_thing, TP_iou_thing,  'result/pretrained/Panoptic_thing')
        # total_ap += ap_thing
        # print(f'AP_{threshold*100} = {ap_thing}' )
        PQ_all_A, SQ_all_A, RQ_all_A = compute_eval2_stuff(N_GT_stuff, N_FN_stuff, N_TP_stuff, N_FP_stuff, TP_iou_stuff, 'result/pretrained/Panoptic_all')
        PQ_all_B, SQ_all_B, RQ_all_B = compute_eval2_stuff(N_GT_stuff, N_FN_all, N_TP_all, N_FP_all, TP_iou_all, 'result/pretrained/Panoptic_all')


        # print(f'AP_thing={ap_thing}')
        # # print(f'AP_stuff={ap_stuff}')
        print('Type A:')
        print(f'PQ={PQ_all_A}')
        print(f'SQ={SQ_all_A}')
        print(f'RQ={RQ_all_A}')
        print()
        print(f'PQ_thing={PQ_thing}')
        print(f'SQ_thing={SQ_thing}')
        print(f'RQ_thing={RQ_thing}')
        print()
        print(f'PQ_stuff={PQ_stuff}')
        print(f'SQ_stuff={SQ_stuff}')
        print(f'RQ_stuff={RQ_stuff}')
        print()
        print('Type B:')
        print(f'PQ={PQ_all_B}')
        print(f'SQ={SQ_all_B}')
        print(f'RQ={RQ_all_B}')
        print()
        print(f'PQ_thing={PQ_thing}')
        print(f'SQ_thing={SQ_thing}')
        print(f'RQ_thing={RQ_thing}')
        print()
        print(f'PQ_stuff={PQ_stuff_B}')
        print(f'SQ_stuff={SQ_stuff_B}')
        print(f'RQ_stuff={RQ_stuff_B}')

        TP_scores = np.concatenate(TP_scores)
        FP_scores = np.concatenate(FP_scores)
        ap, mrec_thing, mpre_thing = compute_eval2(N_GT, TP_scores, FP_scores, 'result/pretrained/Panoptic_thing')
        total_ap += ap
        print(f'AP_{threshold*100} = {ap}' )


        # print()
        # print(f'PQ_stuff={sum(TP_iou_stuff) / (N_TP_stuff + N_FN_stuff/2 + N_FP_stuff/2)}')
        # print(sum(TP_iou_stuff))
        # break

    print(f"Confusion_matrix:{confusion_matrix}")
    print('precision: ', precision)
    print('sensitivity: ', sensitivity)
    print('F1: ', F1)

    print(f'AP = {total_ap / 10}')



if __name__ == '__main__':

    #BW
    # panoptic_gt_path = 'data/CutImage/panoptic_label/T5'
    # sem_gt_path = 'data/CutImage/sem_seg/T5'
    # pred_path = 'result/pretrained/results_BWPanoptic/T5/FL_ff1'
    # instance_gt_path = 'data/CutImage/instance_label'

    # PA
    panoptic_gt_path = 'data/PA/panoptic_label/T5'
    instance_gt_path = 'data/PA/instance_label'
    sem_gt_path = 'data/PA/sem_seg/T5'
    pred_path = 'result/pretrained/results_PAPanoptic/T5/FL_e'

    run_eval_pkl(panoptic_gt_path, instance_gt_path, sem_gt_path, pred_path)
    # run_eval_np(gt_path, pred_path)
