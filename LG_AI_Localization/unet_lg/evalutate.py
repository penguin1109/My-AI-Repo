import json
import numpy as np
import pycocotools.mask as mask

def f1_with_iou(gt, pr, th=0.01):
    tp_iou = []
    tp = []
    fp = []
    fn = []
    
    for img in gt['images']:
        gt_img = [i for i in gt['annotations'] if i['image_id'] == img['id']]
        pr_img = [i for i in pr['annotations'] if i['image_id'] == img['id']]
        # 해당 GT에 대한 예측이 있는 경우
        if len(pr_img) > 0:
            # pr_img = [i for i in pr['annotations'] if i['image_id'] == pr['images'][0]['id']] # for sample test
            ious = [mask.iou([i['segmentation']], [j['segmentation']], [0]) for i in gt_img for j in pr_img]
            ioumat = np.array(ious).reshape(len(gt_img), -1) # gt_dim:0, pr_dim:1
            
            # pr을 iou가 최대인 gt에 할당
            np.argmax(ioumat, axis=0)
            ioumat = ioumat * (ioumat.max(axis=0, keepdims=True) == ioumat)
            
            # TP_IoU / FP / FN
            max_vals = np.amax(ioumat, axis=1)
            tp_iou.extend([i for i in max_vals if i != 0])
            tp.append(sum(max_vals != 0))
            fp.extend([sum(i > th) -1 for i in ioumat if sum(i > th) >= 2])
            fn.append(sum(max_vals == 0))
        # 해당 GT에 대한 예측이 없는 경우 모든 object를 FN에 추가
        else: 
            fn.append(len(gt_img))
    
    tp_iou = np.sum(tp_iou)
    tp = np.sum(tp)
    fp = np.sum(fp)
    fn = np.sum(fn)
    
    precision = tp_iou / (tp + fp)
    recall = tp_iou / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    
    return f1_score