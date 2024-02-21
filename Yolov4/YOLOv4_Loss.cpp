#include "YOLOv4_Loss.h"



std::vector<torch::Tensor> YOLOv4_LossImpl::forward(std::vector<torch::Tensor> pred, std::vector<torch::Tensor> gt_boxes, std::vector<torch::Tensor> gt_labels)
{
	int batch_size = gt_boxes.size();

	torch::Tensor pred_targets_l = pred[0];
	torch::Tensor pred_targets_m = pred[1];
	torch::Tensor pred_targets_s = pred[2];

	int size_l = pred_targets_l.sizes()[1];
	int size_m = pred_targets_m.sizes()[1];
	int size_s = pred_targets_s.sizes()[1];

	std::vector<torch::Tensor> vpred_targets_l = this->coder->split_preds(pred_targets_l);
	std::vector<torch::Tensor> vpred_targets_m = this->coder->split_preds(pred_targets_m);
	std::vector<torch::Tensor> vpred_targets_s = this->coder->split_preds(pred_targets_s);

	torch::Tensor pred_txty_l		= vpred_targets_l[0];
	torch::Tensor pred_twth_l		= vpred_targets_l[1];
	torch::Tensor pred_objectness_l = vpred_targets_l[2];
	torch::Tensor pred_classes_l	= vpred_targets_l[3];

	torch::Tensor pred_txty_m		= vpred_targets_m[0];
	torch::Tensor pred_twth_m		= vpred_targets_m[1];
	torch::Tensor pred_objectness_m = vpred_targets_m[2];
	torch::Tensor pred_classes_m	= vpred_targets_m[3];

	torch::Tensor pred_txty_s		= vpred_targets_s[0];
	torch::Tensor pred_twth_s		= vpred_targets_s[1];
	torch::Tensor pred_objectness_s = vpred_targets_s[2];
	torch::Tensor pred_classes_s	= vpred_targets_s[3];

	torch::Tensor pred_bbox_l = torch::cat({ pred_txty_l, pred_twth_l }, -1);
	torch::Tensor pred_bbox_m = torch::cat({ pred_txty_m, pred_twth_m }, -1);
	torch::Tensor pred_bbox_s = torch::cat({ pred_txty_s, pred_twth_s }, -1);

	std::vector<torch::Tensor> vpred_bbox = this->coder->decode(std::vector<torch::Tensor>({ pred_bbox_l, pred_bbox_m, pred_bbox_s }));
	pred_bbox_l = vpred_bbox[0];
	pred_bbox_m = vpred_bbox[1];
	pred_bbox_s = vpred_bbox[2];

	torch::Tensor pred_x1y1x2y2_l = util::cxcy_to_xy(pred_bbox_l);
	torch::Tensor pred_x1y1x2y2_m = util::cxcy_to_xy(pred_bbox_m);
	torch::Tensor pred_x1y1x2y2_s = util::cxcy_to_xy(pred_bbox_s);

	std::vector<torch::Tensor> various_targets = this->coder->build_target(gt_boxes, gt_labels);

	torch::Tensor gt_prop_txty_l	= various_targets[0];
	torch::Tensor gt_twth_l			= various_targets[1];
	torch::Tensor gt_objectness_l	= various_targets[2];
	torch::Tensor gt_classes_l		= various_targets[3];
	torch::Tensor ignore_mask_l		= various_targets[4];

	torch::Tensor gt_prop_txty_m	= various_targets[5];
	torch::Tensor gt_twth_m			= various_targets[6];
	torch::Tensor gt_objectness_m	= various_targets[7];
	torch::Tensor gt_classes_m		= various_targets[8];
	torch::Tensor ignore_mask_m		= various_targets[9];

	torch::Tensor gt_prop_txty_s	= various_targets[10];
	torch::Tensor gt_twth_s			= various_targets[11];
	torch::Tensor gt_objectness_s	= various_targets[12];
	torch::Tensor gt_classes_s		= various_targets[13];
	torch::Tensor ignore_mask_s		= various_targets[14];

	torch::Tensor gt_bbox_l = torch::cat({ gt_prop_txty_l, gt_twth_l }, -1);
	torch::Tensor gt_bbox_m = torch::cat({ gt_prop_txty_m, gt_twth_m }, -1);
	torch::Tensor gt_bbox_s = torch::cat({ gt_prop_txty_s, gt_twth_s }, -1);

	std::vector<torch::Tensor> vgt_bbox = this->coder->decode(std::vector<torch::Tensor>({ gt_bbox_l, gt_bbox_m, gt_bbox_s }));
	gt_bbox_l = vgt_bbox[0];
	gt_bbox_m = vgt_bbox[1];
	gt_bbox_s = vgt_bbox[2];

	torch::Tensor gt_x1y1x2y2_l = util::cxcy_to_xy(gt_bbox_l);
	torch::Tensor gt_x1y1x2y2_m = util::cxcy_to_xy(gt_bbox_m);
	torch::Tensor gt_x1y1x2y2_s = util::cxcy_to_xy(gt_bbox_s);

	torch::Tensor xy_loss_l = this->giou_loss(gt_x1y1x2y2_l, pred_x1y1x2y2_l) * gt_objectness_l.squeeze(-1);
	torch::Tensor wh_loss_l = this->giou_loss(gt_x1y1x2y2_l, pred_x1y1x2y2_l) * gt_objectness_l.squeeze(-1);

	torch::Tensor obj_loss_l = gt_objectness_l * this->bce->forward(pred_objectness_l, gt_objectness_l);
	torch::Tensor no_obj_loss_l = (1 - gt_objectness_l) * this->bce->forward(pred_objectness_l, gt_objectness_l) * ignore_mask_l.unsqueeze(-1);
	torch::Tensor classes_loss_l = gt_objectness_l * this->bce->forward(pred_classes_l, gt_classes_l);

	torch::Tensor xy_loss_m = this->giou_loss(gt_x1y1x2y2_m, pred_x1y1x2y2_m) * gt_objectness_m.squeeze(-1);
	torch::Tensor wh_loss_m = this->giou_loss(gt_x1y1x2y2_m, pred_x1y1x2y2_m) * gt_objectness_m.squeeze(-1);

	torch::Tensor obj_loss_m = gt_objectness_m * this->bce->forward(pred_objectness_m, gt_objectness_m);
	torch::Tensor no_obj_loss_m = (1 - gt_objectness_m) * this->bce->forward(pred_objectness_m, gt_objectness_m) * ignore_mask_m.unsqueeze(-1);
	torch::Tensor classes_loss_m = gt_objectness_m * this->bce->forward(pred_classes_m, gt_classes_m);

	torch::Tensor xy_loss_s = this->giou_loss(gt_x1y1x2y2_s, pred_x1y1x2y2_s) * gt_objectness_s.squeeze(-1);
	torch::Tensor wh_loss_s = this->giou_loss(gt_x1y1x2y2_s, pred_x1y1x2y2_s) * gt_objectness_s.squeeze(-1);

	torch::Tensor obj_loss_s = gt_objectness_s * this->bce->forward(pred_objectness_s, gt_objectness_s);
	torch::Tensor no_obj_loss_s = (1 - gt_objectness_s) * this->bce->forward(pred_objectness_s, gt_objectness_s) * ignore_mask_s.unsqueeze(-1);
	torch::Tensor classes_loss_s = gt_objectness_s * this->bce->forward(pred_classes_s, gt_classes_s);


	// torch::Tensor xy_loss = 5 * (xy_loss_l.sum() + xy_loss_m.sum() + xy_loss_s.sum()) / batch_size;
	// torch::Tensor xy_loss = 5 * (xy_loss_l.sum() + xy_loss_m.sum() + xy_loss_s.sum()) / batch_size;
	torch::Tensor xy_loss = 2.5 * (xy_loss_l.sum() + xy_loss_m.sum() + xy_loss_s.sum()) / batch_size;
	torch::Tensor wh_loss = 2.5 * (wh_loss_l.sum() + wh_loss_m.sum() + wh_loss_s.sum()) / batch_size;
	torch::Tensor obj_loss = 1 * (obj_loss_l.sum() + obj_loss_m.sum() + obj_loss_m.sum()) / batch_size;
	torch::Tensor no_obj_loss = 0.5 * (no_obj_loss_l.sum() + no_obj_loss_m.sum() + no_obj_loss_s.sum()) / batch_size;
	torch::Tensor cls_loss = 1 * (classes_loss_l.sum() + classes_loss_m.sum() + classes_loss_s.sum()) / batch_size;

	torch::Tensor total_loss = xy_loss + wh_loss + obj_loss + no_obj_loss + cls_loss;

	return std::vector<torch::Tensor>({ total_loss, xy_loss, wh_loss, obj_loss, no_obj_loss, cls_loss });
}

torch::Tensor YOLOv4_LossImpl::giou_loss(torch::Tensor boxes1, torch::Tensor boxes2)
{
	torch::Tensor boxes1_area =
		(boxes1.index({ torch::indexing::Ellipsis, 2 }) - boxes1.index({ torch::indexing::Ellipsis, 0 })) *
		(boxes1.index({ torch::indexing::Ellipsis, 3 }) - boxes1.index({ torch::indexing::Ellipsis, 1 }));

	torch::Tensor boxes2_area =
		(boxes2.index({ torch::indexing::Ellipsis, 2 }) - boxes2.index({ torch::indexing::Ellipsis, 0 })) *
		(boxes2.index({ torch::indexing::Ellipsis, 3 }) - boxes2.index({ torch::indexing::Ellipsis, 1 }));

	torch::Tensor inter_left_up =
		torch::max(
			boxes1.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) }),
			boxes2.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) })
		);

	torch::Tensor inter_right_down =
		torch::min(
			boxes1.index({torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None)}),
			boxes2.index({torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None)})
		);

	torch::Tensor inter_section = 
		torch::max(inter_right_down - inter_left_up, torch::zeros_like(inter_right_down));
	torch::Tensor inter_area = 
		inter_section.index({ torch::indexing::Ellipsis, 0 }) * inter_section.index({ torch::indexing::Ellipsis, 1 });
	torch::Tensor union_area = boxes1_area + boxes2_area - inter_area;
	torch::Tensor ious = 1.0 * (inter_area / union_area);

	torch::Tensor outer_left_up =
		torch::min(
			boxes1.index({torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2)}),
			boxes2.index({torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2)})
		);

	torch::Tensor outer_right_down =
		torch::max(
			boxes1.index({torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None)}),
			boxes2.index({torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None)})
		);

	torch::Tensor outer_section =
		torch::max(outer_right_down - outer_left_up, torch::zeros_like(inter_right_down));

	torch::Tensor outer_area =
		outer_section.index({ torch::indexing::Ellipsis, 0 }) * outer_section.index({ torch::indexing::Ellipsis, 1 });

	torch::Tensor giou = ious - (outer_area - union_area) / outer_area;

	torch::Tensor giou_loss = 1 - giou;

	return giou_loss;
}