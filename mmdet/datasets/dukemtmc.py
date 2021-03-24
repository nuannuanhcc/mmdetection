from .coco import CocoDataset
from .builder import DATASETS
import numpy as np
from collections import defaultdict
from ..core.evaluation.bbox_overlaps import bbox_overlaps
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
import os.path as osp
import time

@DATASETS.register_module
class DukemtmcDataset(CocoDataset):
    CLASSES = ('person',)

    def map_class_id_to_class_name(self, class_id):
        return self.CLASSES[class_id]

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)

                if self.with_reid:
                    if self.is_query or self.is_test:
                        gt_labels.append([self.cat2label[ann['category_id']], ann['pid']])
                    else:
                        # gt_labels.append([self.cat2label[ann['category_id']], ann['pid'], ann['image_id'], ann['id'],
                        #                   ann['id_labeled']])
                        ann_pid = ann['pid'] if ann['pid'] == -1 else ann['pid'] + 5532 + 932
                        ann_image_id = ann['image_id'] + 11206 + 5704 + 902 + 1062
                        ann_id = ann['id'] + 55260 + 18048 + 1826 + 7412
                        ann_id_labeled = ann['id_labeled'] if ann['id_labeled'] == -1 else ann['id_labeled'] + 15080 + 14907
                        gt_labels.append(
                            [self.cat2label[ann['category_id']], ann_pid, ann_image_id, ann_id, ann_id_labeled])
                else:
                    gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def evaluate(self, predictions, dataset):
        if self.with_reid:
            result = self.evaluate_reid(predictions, dataset)
        else:
            result = self.evaluate_detection(predictions)
        return result

    def evaluate_reid(self, predictions, dataset, gallery_size=100):
        dataset_test, dataset_query = dataset
        predictions_test, predictions_query = predictions
        query_img_list = [img['file_name'] for img in dataset_query.data_infos]
        query_feats = [gt[-1] for gt in predictions_query]

        pred_boxlists = []
        gt_boxlists = []
        test_img_list = []
        pred_feats = []

        for image_id, prediction in enumerate(tqdm(predictions_test)):
            if len(prediction) == 0:
                continue
            img_name = dataset_test.data_infos[image_id]["file_name"]
            test_img_list.append(img_name)

            pred_feats.append(prediction[-1])
            pred_boxlists.append(prediction[0][0][0])

            gt_boxlist = dataset_test.get_ann_info(image_id)['bboxes']
            gt_boxlists.append(gt_boxlist)

        result = self.eval_reid_sysu(
            pred_feats=pred_feats,
            query_feats=query_feats,
            test_img_list=test_img_list,
            query_img_list=query_img_list,
            query_boxlists=predictions_query,
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=0.5,
            use_07_metric=False,
            gallery_size=gallery_size,
        )

        topk = [1, 3, 5, 10]
        result_str = "\n##################################################\n"
        result_str += '###   gallery_size = {}   ##################'.format(gallery_size)
        result_str += '\nmodel: {}\n'.format(result['model'])
        result_str += "Detection_Recall: {:.4f}\n".format(result["Detection_Recall"])
        result_str += "Detection_Precision: {:.4f}\n".format(result["Detection_Precision"])
        result_str += "Detection_mean_Avg_Precision: {:.4f}\n".format(result["Detection_mean_Avg_Precision"])
        result_str += "ReID_Recall: {:.4f}  ".format(result["ReID_Recall"])
        result_str += "(ReID_Recall_Ideal: {:.4f})\n".format(result["ReID_Recall_Ideal"])
        result_str += "ReID_mean_Avg_Precision: {:.4f}  ".format(result["ReID_mean_Avg_Precision"])
        result_str += "(ReID_mean_Avg_Precision_Ideal: {:.4f})\n".format(result["ReID_mean_Avg_Precision_Ideal"])
        for i, k in enumerate(topk):
            result_str += '  Top-{:2d} = {:.2%} (The ideal top-{:2d} = {:.2%})\n'.format(k, result["CMC"][i], k,
                                                                                         result["CMC_Ideal"][i])
        result_str += "##################################################\n"
        print(result_str)
        return result_str

    def evaluate_detection(self, predictions, iou_thr=0.5):
        pred_boxlists = []
        gt_boxlists = []
        for image_id, prediction in enumerate(predictions[0]):
            prediction = prediction[0]  #  TODO n_box * 5
            gt_boxlist = self.get_ann_info(image_id)['bboxes']
            if len(prediction) == 0:
                continue
            pred_boxlists.append(prediction)
            gt_boxlists.append(gt_boxlist)

        result = self.eval_detection_sysu(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=iou_thr,
            use_07_metric=False,
        )
        result_str = "\n##################################################\n"
        result_str += "mAP: {:.4f}\n".format(result["map"])
        for i, ap in enumerate(result["ap"]):
            if i == 0:  # skip background
                continue
            result_str += "{:<8}: {:.4f}\n".format(
                self.CLASSES[i-1], ap
            )
        result_str += "##################################################\n"
        print(result_str)
        return result_str

    def eval_detection_sysu(self, pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
        """Evaluate on voc dataset.
        Args:
            pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
            gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
            iou_thresh: iou thresh
            use_07_metric: boolean
        Returns:
            dict represents the results
        """
        assert len(gt_boxlists) == len(
            pred_boxlists
        ), "Length of gt and pred lists need to be same."
        prec, rec = self.calc_detection_sysu_prec_rec(
            pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
        )
        ap = self.calc_detection_sysu_ap(prec, rec, use_07_metric=use_07_metric)
        return {"ap": ap, "map": np.nanmean(ap)}

    def calc_detection_sysu_prec_rec(self, gt_boxlists, pred_boxlists, iou_thresh=0.5):
        """Calculate precision and recall based on evaluation code of PASCAL VOC.
        This function calculates precision and recall of
        predicted bounding boxes obtained from a dataset which has :math:`N`
        images.
        The code is based on the evaluation code used in PASCAL VOC Challenge.
       """
        n_pos = defaultdict(int)
        score = defaultdict(list)
        match = defaultdict(list)
        for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
            pred_bbox = pred_boxlist[:, :4]
            pred_label = np.ones(pred_bbox.shape[0])  # TODO
            pred_score = pred_boxlist[:, -1]
            gt_bbox = gt_boxlist
            gt_label = np.ones(gt_bbox.shape[0])  # TODO
            gt_difficult = np.zeros(gt_bbox.shape[0])  # TODO

            for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
                pred_mask_l = pred_label == l
                pred_bbox_l = pred_bbox[pred_mask_l]
                pred_score_l = pred_score[pred_mask_l]
                # sort by score
                order = pred_score_l.argsort()[::-1]
                pred_bbox_l = pred_bbox_l[order]
                pred_score_l = pred_score_l[order]

                gt_mask_l = gt_label == l
                gt_bbox_l = gt_bbox[gt_mask_l]
                gt_difficult_l = gt_difficult[gt_mask_l]

                n_pos[l] += np.logical_not(gt_difficult_l).sum()
                score[l].extend(pred_score_l)

                if len(pred_bbox_l) == 0:
                    continue
                if len(gt_bbox_l) == 0:
                    match[l].extend((0,) * pred_bbox_l.shape[0])
                    continue

                # VOC evaluation follows integer typed bounding boxes.
                pred_bbox_l = pred_bbox_l.copy()
                pred_bbox_l[:, 2:] += 1
                gt_bbox_l = gt_bbox_l.copy()
                gt_bbox_l[:, 2:] += 1
                iou = bbox_overlaps(pred_bbox_l, gt_bbox_l)
                gt_index = iou.argmax(axis=1)
                # set -1 if there is no matching ground truth
                gt_index[iou.max(axis=1) < iou_thresh] = -1
                del iou

                selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:
                        if gt_difficult_l[gt_idx]:
                            match[l].append(-1)
                        else:
                            if not selec[gt_idx]:
                                match[l].append(1)
                            else:
                                match[l].append(0)
                        selec[gt_idx] = True
                    else:
                        match[l].append(0)

        n_fg_class = max(n_pos.keys()) + 1
        prec = [None] * n_fg_class
        rec = [None] * n_fg_class

        for l in n_pos.keys():
            score_l = np.array(score[l])
            match_l = np.array(match[l], dtype=np.int8)

            order = score_l.argsort()[::-1]
            match_l = match_l[order]

            tp = np.cumsum(match_l == 1)
            fp = np.cumsum(match_l == 0)

            # If an element of fp + tp is 0,
            # the corresponding element of prec[l] is nan.
            prec[l] = tp / (fp + tp)
            # If n_pos[l] is 0, rec[l] is None.
            if n_pos[l] > 0:
                rec[l] = tp / n_pos[l]

        return prec, rec

    def calc_detection_sysu_ap(self, prec, rec, use_07_metric=False):
        """Calculate average precisions based on evaluation code of PASCAL VOC.
        This function calculates average precisions
        from given precisions and recalls.
        The code is based on the evaluation code used in PASCAL VOC Challenge.
        Args:
            prec (list of numpy.array): A list of arrays.
                :obj:`prec[l]` indicates precision for class :math:`l`.
                If :obj:`prec[l]` is :obj:`None`, this function returns
                :obj:`numpy.nan` for class :math:`l`.
            rec (list of numpy.array): A list of arrays.
                :obj:`rec[l]` indicates recall for class :math:`l`.
                If :obj:`rec[l]` is :obj:`None`, this function returns
                :obj:`numpy.nan` for class :math:`l`.
            use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
                for calculating average precision. The default value is
                :obj:`False`.
        Returns:
            ~numpy.ndarray:
            This function returns an array of average precisions.
            The :math:`l`-th value corresponds to the average precision
            for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
            :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
        """

        n_fg_class = len(prec)
        ap = np.empty(n_fg_class)
        for l in range(n_fg_class):
            if prec[l] is None or rec[l] is None:
                ap[l] = np.nan
                continue

            if use_07_metric:
                # 11 point metric
                ap[l] = 0
                for t in np.arange(0.0, 1.1, 0.1):
                    if np.sum(rec[l] >= t) == 0:
                        p = 0
                    else:
                        p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                    ap[l] += p / 11
            else:
                # correct AP calculation
                # first append sentinel values at the end
                mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
                mrec = np.concatenate(([0], rec[l], [1]))

                mpre = np.maximum.accumulate(mpre[::-1])[::-1]

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec
                ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def calc_reid_sysu_topk(self, pred_feats, query_feats, test_img_list, query_img_list, query_boxlists, pred_boxlists, gt_boxlists, topk,
                            det_thresh=0.5, iou_thresh=0.5, gallery_size=100):

        # assert len(test_img_list) == 6978, "Length of test_img lists need to be 6978, but it is {}.".format(len(test_img_list))
        assert len(query_img_list) == 2900, "Length of query_img lists need to be 2900, but it is {}.".format(
            len(query_img_list))
        assert len(query_boxlists) == 2900, "Length of query_boxlists lists need to be 2900, but it is {}.".format(
            len(query_boxlists))
        assert len(gt_boxlists) == len(pred_boxlists), "Length of gt and pred lists need to be same."
        assert isinstance(topk, (list, tuple)), "topk must be a list or tuple !"


        ############################# path #############################
        annotation_dir = './data/sysu/SIPN_annotation/'

        test_all_file = 'testAllDF.csv'
        query_file = 'queryDF.csv'
        q_to_g_file = 'q_to_g' + str(gallery_size) + 'DF.csv'

        test_all = pd.read_csv(osp.join(annotation_dir, test_all_file))
        query_boxes = pd.read_csv(osp.join(annotation_dir, query_file))
        queries_to_galleries = pd.read_csv(osp.join(annotation_dir, q_to_g_file))

        test_all, query_boxes = delta_to_coordinates(test_all, query_boxes)
        ################################ initial ################################
        test_imnames = test_img_list
        query_imnames = query_img_list
        gallery_det = pred_boxlists
        gallery_feat = pred_feats
        probe_feat = [p_f.squeeze() for p_f in query_feats]

        use_full_set = gallery_size == -1
        df = test_all.copy()

        # ====================formal=====================
        name_to_det_feat = {}
        for name, det, feat in zip(test_imnames, gallery_det, gallery_feat):
            scores = det[:, 4].ravel()
            inds = np.where(scores >= det_thresh)[0]
            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds])

        # # =====================debug=====================
        # f = open('name_to_det_feat.pkl', 'rb+')
        # name_to_det_feat = pickle.load(f)
        # # ======================end======================

        rec = []
        rec_ideal = []
        aps = []
        aps_ideal = []  ### Ideal situation
        accs = []
        accs_ideal = []  ### Ideal situation
        topk = topk
        # ret  # TODO: save json
        for i in range(len(probe_feat)):
            pid = query_boxes.loc[i, 'pid']
            num_g = query_boxes.loc[i, 'num_g']
            y_true, y_score = [], []
            y_true_ideal = []  ### Ideal situation
            imgs, rois = [], []
            count_gt, count_tp = 0, 0
            count_tp_ideal = 0  ### Ideal situation
            # Get L2-normalized feature vector
            feat_p = probe_feat[i].ravel()
            # Ignore the probe image
            start = time.time()
            probe_imname = queries_to_galleries.iloc[i, 0]
            probe_gt = []
            tested = set([probe_imname])
            # 1. Go through the gallery samples defined by the protocol
            for g_i in range(1, gallery_size + 1):

                gallery_imname = queries_to_galleries.iloc[i, g_i]
                if g_i <= num_g:
                    gt = df.query('imname==@gallery_imname and pid==@pid')
                    gt = gt.loc[:, 'x1': 'y2'].values.ravel()
                else:
                    gt = np.array([])
                count_gt += (gt.size > 0)
                # compute distance between probe and gallery dets
                if gallery_imname not in name_to_det_feat: continue

                det, feat_g = name_to_det_feat[gallery_imname]
                # get L2-normalized feature matrix NxD
                assert feat_g.size == np.prod(feat_g.shape[:2])
                feat_g = feat_g.reshape(feat_g.shape[:2])
                # compute cosine similarities
                sim = feat_g.dot(feat_p).ravel()
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                label_ideal = np.zeros(len(sim), dtype=np.int32)  ### Ideal situation
                if gt.size > 0:
                    w, h = gt[2] - gt[0], gt[3] - gt[1]
                    probe_gt.append({'img': str(gallery_imname), 'roi': list(gt.astype('float'))})
                    iou_thresh = min(.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]

                    label_ideal[0] = 1  # Ideally
                    count_tp_ideal += 1  # Ideally
                    # label[0] = 1
                    # count_tp += 1
                    # """
                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if _compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                    # """

                y_true.extend(list(label))
                y_true_ideal.extend(list(label_ideal))  ### Ideal situation
                y_score.extend(list(sim))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))
                tested.add(gallery_imname)
            # 2. Go through the remaining gallery images if using full set
            if use_full_set:
                pass  # TODO
            # 3. Compute AP for this probe (need to scale by recall rate)
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            y_true_ideal = np.asarray(y_true_ideal)  ### Ideal situation
            assert count_tp <= count_gt
            assert count_tp_ideal <= count_gt  ### Ideal situation
            if count_gt == 0:
                print(probe_imname, i)
                break
            recall_rate = count_tp * 1.0 / count_gt
            recall_rate_ideal = count_tp_ideal * 1.0 / count_gt  ### Ideal situation
            ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
            ap_ideal = 0 if count_tp_ideal == 0 else average_precision_score(y_true_ideal,
                                                                             y_score) * recall_rate_ideal  ### Ideal situation
            rec.append(recall_rate)
            rec_ideal.append(recall_rate_ideal)
            aps.append(ap)
            aps_ideal.append(ap_ideal)  ### Ideal situation
            inds = np.argsort(y_score)[::-1]
            y_score = y_score[inds]
            y_true = y_true[inds]
            y_true_ideal = y_true_ideal[inds]  ### Ideal situation


            """
            num = 0.0
            ap = []
            for i, y_t in enumerate(y_true):
                if y_t == 1:
                    num += 1
                    ap.append(num / (int(i) + 1))
            aps.append(np.mean(ap))
            """

            accs.append([min(1, sum(y_true[:k])) for k in topk])
            accs_ideal.append([min(1, sum(y_true_ideal[:k])) for k in topk])  ### Ideal situation
            # compute time cost
            end = time.time()

        return (rec, aps, accs), (rec_ideal, aps_ideal, accs_ideal)

    def eval_reid_sysu(self, pred_feats, query_feats, test_img_list, query_img_list, query_boxlists, pred_boxlists, gt_boxlists, iou_thresh=0.5,
                       use_07_metric=False, gallery_size=100):
        """Evaluate on voc dataset.
        Args:
            pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
            gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
            iou_thresh: iou thresh
            use_07_metric: boolean
        Returns:
            dict represents the results
        """
        # assert len(test_img_list) == 6978, "Length of test_img lists need to be 6978, but it is {}.".format(len(test_img_list))
        assert len(query_img_list) == 2900, "Length of query_img lists need to be 2900, but it is {}.".format(
            len(query_img_list))
        assert len(query_boxlists) == 2900, "Length of query_boxlists lists need to be 2900, but it is {}.".format(
            len(query_boxlists))
        assert len(gt_boxlists) == len(pred_boxlists), "Length of gt and pred lists need to be same."

        result = {}
        result.update({'model': 'model_0'})

        prec, rec = self.calc_detection_sysu_prec_rec(pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh)
        Detection_Precision = 100 * np.nanmean(prec[1])
        Detection_Recall = 100 * np.nanmean(rec[1])
        result.update({'Detection_Precision': Detection_Precision})
        result.update({'Detection_Recall': Detection_Recall})
        ap = self.calc_detection_sysu_ap(prec, rec, use_07_metric=use_07_metric)
        Detection_mean_Avg_Precision = np.nanmean(ap)
        result.update({'Detection_mean_Avg_Precision': Detection_mean_Avg_Precision})

        ##################  Re-ID   ###################
        topk = [1, 3, 5, 10]
        # CMC = []
        real_result, ideal_result = self.calc_reid_sysu_topk(pred_feats, query_feats, test_img_list, query_img_list, query_boxlists, pred_boxlists,
                                                        gt_boxlists, topk, det_thresh=0.5, iou_thresh=0.5,
                                                        gallery_size=gallery_size)
        rec, aps, accs = real_result
        rec_ideal, aps_ideal, accs_ideal = ideal_result
        ReID_Recall = np.nanmean(rec)
        ReID_Recall_Ideal = np.nanmean(rec_ideal)
        ReID_mean_Avg_Precision = np.nanmean(aps)
        ReID_mean_Avg_Precision_Ideal = np.nanmean(aps_ideal)
        result.update({'ReID_Recall': ReID_Recall})
        result.update({'ReID_Recall_Ideal': ReID_Recall_Ideal})
        result.update({'ReID_mean_Avg_Precision': ReID_mean_Avg_Precision})
        result.update({'ReID_mean_Avg_Precision_Ideal': ReID_mean_Avg_Precision_Ideal})
        accs = np.mean(accs, axis=0)
        accs_ideal = np.mean(accs_ideal, axis=0)
        result.update({'CMC': accs})
        result.update({'CMC_Ideal': accs_ideal})
        return result

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

def delta_to_coordinates(test_all, query_boxes):
    """change `del_x` and `del_y` to `x2` and `y2` for testing set"""
    test_all['del_x'] += test_all['x1']
    test_all['del_y'] += test_all['y1']
    test_all.rename(columns={'del_x': 'x2', 'del_y': 'y2'}, inplace=True)
    query_boxes['del_x'] += query_boxes['x1']
    query_boxes['del_y'] += query_boxes['y1']
    query_boxes.rename(columns={'del_x': 'x2', 'del_y': 'y2'}, inplace=True)
    return test_all, query_boxes
