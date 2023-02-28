import numpy as np
import cv2
from matplotlib import pyplot as plt
# from pyrsistent import T
IOU_THRESH = 0.5  # for matching


def compute_eval1(N_GT: int, TP_scores: np.ndarray, FP_scores: np.ndarray):
    N_TP, N_FP = TP_scores.shape[0], FP_scores.shape[0]
    N_DE = N_TP + N_FP
    scores = np.concatenate((TP_scores, FP_scores))
    flagTP = np.zeros(N_DE, np.bool)
    flagTP[:N_TP] = True
    # sort
    idx = np.argsort(scores)[::-1]
    scores, flagTP = scores[idx], flagTP[idx]
    prec, rec = [], []
    now_TP, now_FP = 0, 0
    for i in range(N_DE):
        if flagTP[i]:
            now_TP += 1
        else:
            now_FP += 1
        prec.append(now_TP / (i + 1))
        rec.append(now_TP / N_GT)
    # VOC AP (this code is copied from github mAP)
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    # print(ap)
    return ap, mrec, mpre

def compute_eval2(N_GT, TP_scores, FP_scores, out_path: str):
    N_TP, N_FP = TP_scores.shape[0], FP_scores.shape[0]
    N_DE = N_TP + N_FP
    scores = np.concatenate((TP_scores, FP_scores))
    flagTP = np.zeros(N_DE, np.bool)
    flagTP[:N_TP] = True
    # sort
    idx = np.argsort(scores)[::-1]
    scores, flagTP = scores[idx], flagTP[idx]
    prec, rec = [], []
    now_TP, now_FP = 0, 0
    for i in range(N_DE):
        if flagTP[i]:
            now_TP += 1
        else:
            now_FP += 1
        prec.append(now_TP / (i + 1))
        rec.append(now_TP / N_GT)
    # VOC AP (this code is copied from github mAP)
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    # print(ap)
    text = "{0:.2f}%".format(ap * 100) + " = " + 'person' + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
    # plot.............................(this code is copied and modified from github mAP)
    plt.figure(facecolor='snow')
    # plt.set_facecolor('snow')
    plt.plot(rec[1:-1], prec[1:-1], '-o')
    # add a new penultimate point to the list (mrec[-2], 0.0)
    # since the last line segment (and respective area) do not affect the AP value
    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mpre[:-1] + [0.0] + [mpre[-1]]
    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
    # set window title
    fig = plt.gcf()  # gcf - get current figure
    fig.canvas.set_window_title('AP')
    # set plot title
    plt.title('class: ' + text)
    #plt.suptitle('This is a somewhat long figure title', fontsize=16)
    # set axis titles
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # optional - set axes
    axes = plt.gca()  # gca - get current axes
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
    # Alternative option -> wait for button to be pressed
    # while not plt.waitforbuttonpress(): pass # wait for key display
    # Alternative option -> normal display
    # plt.show()
    # save the plot
    plt.savefig(out_path, bbox_inches='tight')
    return ap, mrec, mpre

def compute_eval2_thing(N_GT, N_FN, TP_scores, FP_scores, TP_iou, out_path: str):
    N_TP, N_FP = TP_scores.shape[0], FP_scores.shape[0]
    N_DE = N_TP + N_FP
    sum_IOU = np.sum(TP_iou)
    SQ = sum_IOU / N_TP
    # print(TP_iou, sum_IOU)
    RQ = N_TP / (N_TP + N_FP / 2 + N_FN / 2)
    PQ = sum_IOU / (N_TP + N_FP/2 + N_FN/2)
    print("(N_TP, N_FP, N_FN) = ", (N_TP, N_FP, N_FN))
    scores = np.concatenate((TP_scores, FP_scores))
    flagTP = np.zeros(N_DE, np.bool)
    flagTP[:N_TP] = True
    # sort
    idx = np.argsort(scores)[::-1]
    scores, flagTP = scores[idx], flagTP[idx]
    prec, rec = [], []
    now_TP, now_FP = 0, 0
    for i in range(N_DE):
        if flagTP[i]:
            now_TP += 1
        else:
            now_FP += 1
        prec.append(now_TP / (i + 1))
        rec.append(now_TP / N_GT)
    # VOC AP (this code is copied from github mAP)
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    # print(ap)
    text = "{0:.2f}%".format(ap * 100) + " = " + 'person' + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
    # plot.............................(this code is copied and modified from github mAP)
    plt.figure(facecolor='snow')
    # plt.set_facecolor('snow')
    plt.plot(rec[1:-1], prec[1:-1], '-o')
    # add a new penultimate point to the list (mrec[-2], 0.0)
    # since the last line segment (and respective area) do not affect the AP value
    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mpre[:-1] + [0.0] + [mpre[-1]]
    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
    # set window title
    fig = plt.gcf()  # gcf - get current figure
    fig.canvas.set_window_title('AP')
    # set plot title
    plt.title('class: ' + text)
    #plt.suptitle('This is a somewhat long figure title', fontsize=16)
    # set axis titles
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # optional - set axes
    axes = plt.gca()  # gca - get current axes
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
    # Alternative option -> wait for button to be pressed
    # while not plt.waitforbuttonpress(): pass # wait for key display
    # Alternative option -> normal display
    # plt.show()
    # save the plot
    plt.savefig(out_path, bbox_inches='tight')
    return ap, mrec, mpre, PQ, SQ, RQ

def compute_eval2_stuff(N_GT, N_FN, N_TP, flagTP, TP_iou, out_path: str):
    # print(N_GT, N_FN)
    # N_TP = np.argwhere(flagTP == True).shape[0]
    # N_FP = np.argwhere(flagTP == False).shape[0]
    N_FP = flagTP
    # N_FP = 0
    # N_TP, N_FP = TP_scores.shape[0], FP_scores.shape[0]
    # N_DE = N_TP + N_FP
    sum_IOU = np.sum(TP_iou)
    SQ = sum_IOU / N_TP
    RQ = N_TP / (N_TP + N_FP / 2 + N_FN/2)
    PQ = sum_IOU / (N_TP + N_FP/2 + N_FN/2)
    print("(N_TP, N_FP, N_FN) = ", (N_TP, N_FP, N_FN))
    # scores = np.concatenate((TP_scores, FP_scores))
    # flagTP = np.zeros(N_DE, np.bool)
    # flagTP[:N_TP] = True
    # # sort
    # idx = np.argsort(scores)[::-1]
    # scores, flagTP = scores[idx], flagTP[idx]
    # prec, rec = [], []
    # now_TP, now_FP = 0, 0
    # for i in range(N_DE):
    #     if flagTP[i]:
    #         now_TP += 1
    #     else:
    #         now_FP += 1
    #     prec.append(now_TP / (i + 1))
    #     rec.append(now_TP / N_GT)
    # # VOC AP (this code is copied from github mAP)
    # rec.insert(0, 0.0)  # insert 0.0 at begining of list
    # rec.append(1.0)  # insert 1.0 at end of list
    # mrec = rec[:]
    # prec.insert(0, 0.0)  # insert 0.0 at begining of list
    # prec.append(0.0)  # insert 0.0 at end of list
    # mpre = prec[:]
    # for i in range(len(mpre) - 2, -1, -1):
    #     mpre[i] = max(mpre[i], mpre[i + 1])
    # i_list = []
    # for i in range(1, len(mrec)):
    #     if mrec[i] != mrec[i - 1]:
    #         i_list.append(i)  # if it was matlab would be i + 1
    # ap = 0.0
    # for i in i_list:
    #     ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    # # print(ap)
    # text = "{0:.2f}%".format(ap * 100) + " = " + 'person' + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
    # # plot.............................(this code is copied and modified from github mAP)
    # plt.figure(facecolor='snow')
    # # plt.set_facecolor('snow')
    # plt.plot(rec[1:-1], prec[1:-1], '-o')
    # # add a new penultimate point to the list (mrec[-2], 0.0)
    # # since the last line segment (and respective area) do not affect the AP value
    # area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    # area_under_curve_y = mpre[:-1] + [0.0] + [mpre[-1]]
    # plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
    # # set window title
    # fig = plt.gcf()  # gcf - get current figure
    # fig.canvas.set_window_title('AP')
    # # set plot title
    # plt.title('class: ' + text)
    # #plt.suptitle('This is a somewhat long figure title', fontsize=16)
    # # set axis titles
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # # optional - set axes
    # axes = plt.gca()  # gca - get current axes
    # axes.set_xlim([0.0, 1.0])
    # axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
    # # Alternative option -> wait for button to be pressed
    # # while not plt.waitforbuttonpress(): pass # wait for key display
    # # Alternative option -> normal display
    # # plt.show()
    # # save the plot
    # plt.savefig(out_path, bbox_inches='tight')
    return PQ, SQ, RQ


# RotatedRect Format: ((px, py), (w, h), angle)
# angle uses degree unit

def rorc_IOU(rorc1, rorc2, area1=None, area2=None):
    # RotatedRect IOU
    r1 = cv2.rotatedRectangleIntersection(rorc1, rorc2)
    if r1[0] > 0:
        # r1[1] is the unsorted points of a convex hull
        r2 = cv2.convexHull(r1[1])  # sort
        I_area = cv2.contourArea(r2)
    else:
        I_area = 0.0
    area1 = cv2.contourArea(cv2.boxPoints(rorc1)) if area1 is None else area1
    area2 = cv2.contourArea(cv2.boxPoints(rorc2)) if area2 is None else area2
    U_area = area1 + area2 - I_area
    # print(f'{U_area}')
    return I_area / U_area if U_area > 0.0 else 0.0


def mask_IOU(mask1, mask2, area1=None, area2=None):
    I = np.logical_and(mask1, mask2)
    I_area = np.sum(I.astype(int))
    area1 = np.sum(mask1.astype(int)) if area1 is None else area1
    area2 = np.sum(mask2.astype(int)) if area2 is None else area2
    U_area = area1 + area2 - I_area
    return I_area / U_area if U_area > 0.0 else 0.0


def match_masks_thing(masks: np.ndarray, gt_masks: np.ndarray, iou_threshold=0.5):
    flagTP = [False for _ in range(masks.shape[0])]
    gt_used = [False for _ in range(gt_masks.shape[0])]
    TP_iou = [0 for _ in range(masks.shape[0])]
    masks_area = np.sum(masks.astype(int), (1, 2))
    gt_masks_area = np.sum(gt_masks.astype(int), (1, 2))
    for mask_i in range(masks.shape[0]):
        mask = masks[mask_i, ...]
        # print(mask)
        # find max IoU
        max_iou, max_j = 0.0, -1
        for mask_j in range(gt_masks.shape[0]):
            gt_mask = gt_masks[mask_j, ...]
            if gt_used[mask_j]:
                continue
            iou = mask_IOU(mask, gt_mask, masks_area[mask_i], gt_masks_area[mask_j])
            if iou > max_iou:
                max_iou, max_j = iou, mask_j
        if max_j >= 0 and max_iou >= iou_threshold:
            # MATCH
            gt_used[max_j] = True
            flagTP[mask_i] = True
            TP_iou[mask_i] = max_iou
    return flagTP, gt_used, TP_iou

def match_masks_stuff(masks: np.ndarray, gt_masks: np.ndarray):
    flagTP = [False for _ in range(masks.shape[0])]
    gt_used = [False for _ in range(gt_masks.shape[0])]
    TP_iou = [0 for _ in range(masks.shape[0])]
    flagFP = [False for _ in range(masks.shape[0])]
    masks_area = np.sum(masks.astype(int), (1, 2))
    gt_masks_area = np.sum(gt_masks.astype(int), (1, 2))
    for mask_i in range(2, masks.shape[0]):
        mask = masks[mask_i, ...]
        # print(mask) 
        # find max IoU
        max_iou, max_j = 0.0, -1
        for mask_j in range(2, gt_masks.shape[0]):
            gt_mask = gt_masks[mask_j, ...]
            # if gt_used[mask_j]:
            #     continue
            iou = mask_IOU(mask, gt_mask, masks_area[mask_i], gt_masks_area[mask_j])
            
            if iou > max_iou:
                max_iou, max_j = iou, mask_j
        if max_j >= 0 and max_iou >= IOU_THRESH:
            # MATCH
            gt_used[max_j] = True
            flagTP[mask_i] = True
            TP_iou[mask_i] = max_iou

        if masks_area[mask_i] != 0 and max_iou <= IOU_THRESH:
            flagFP[mask_i] = True 

    for i in range(2, gt_masks.shape[0]):
        if gt_masks_area[i] != 0 and masks_area[i] == 0:
            gt_used[i] = True
    

    
    return flagTP, gt_used, flagFP, TP_iou

def match_rorcs(rorcs: list, gt_rorcs: list):
    flagTP = [False for _ in range(len(rorcs))]
    gt_used = [False for _ in range(len(gt_rorcs))]
    for box_i, rorc in enumerate(rorcs):
        # find max IoU
        max_iou, max_j = 0.0, -1
        for box_j, gt_rorc in enumerate(gt_rorcs):
            if gt_used[box_j]:
                continue
            iou = rorc_IOU(rorc, gt_rorc)
            if iou > max_iou:
                max_iou, max_j = iou, box_j
        if max_j >= 0 and max_iou >= IOU_THRESH:
            # MATCH
            gt_used[max_j] = True
            flagTP[box_i] = True
    return flagTP, gt_used
