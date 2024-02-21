#include "YOLOv4_Coder.h"

std::vector<torch::Tensor> YOLOv4_CoderImpl::build_target(std::vector<torch::Tensor> gt_boxes, std::vector<torch::Tensor> gt_labels, bool IT)
{
	int batch_size = gt_labels.size();

	torch::Tensor ignore_mask_l = torch::zeros({ batch_size, 13, 13, 3 }).to(this->device);
	torch::Tensor gt_prop_txty_l = torch::zeros({ batch_size, 13, 13, 3, 2 }).to(this->device);
	torch::Tensor gt_twth_l = torch::zeros({ batch_size, 13, 13, 3, 2 }).to(this->device);
	torch::Tensor gt_objectness_l = torch::zeros({ batch_size, 13, 13, 3, 1 }).to(this->device);
	torch::Tensor gt_classes_l = torch::zeros({ batch_size, 13, 13, 3, this->num_classes }).to(this->device);
	
	torch::Tensor ignore_mask_m = torch::zeros({ batch_size, 26, 26, 3 }).to(this->device);
	torch::Tensor gt_prop_txty_m = torch::zeros({ batch_size, 26, 26, 3, 2 }).to(this->device);
	torch::Tensor gt_twth_m = torch::zeros({ batch_size, 26, 26, 3, 2 }).to(this->device);
	torch::Tensor gt_objectness_m = torch::zeros({ batch_size, 26, 26, 3, 1 }).to(this->device);
	torch::Tensor gt_classes_m = torch::zeros({ batch_size, 26, 26, 3, this->num_classes }).to(this->device);

	torch::Tensor ignore_mask_s = torch::zeros({ batch_size, 52, 52, 3 }).to(this->device);
	torch::Tensor gt_prop_txty_s = torch::zeros({ batch_size, 52, 52, 3, 2 }).to(this->device);
	torch::Tensor gt_twth_s = torch::zeros({ batch_size, 52, 52, 3, 2 }).to(this->device);
	torch::Tensor gt_objectness_s = torch::zeros({ batch_size, 52, 52, 3, 1 }).to(this->device);
	torch::Tensor gt_classes_s = torch::zeros({ batch_size, 52, 52, 3, this->num_classes }).to(this->device);

	torch::Tensor center_anchor_l_ = this->center_anchor_l.clone();
	torch::Tensor corner_anchor_l = util::cxcy_to_xy(center_anchor_l_).view({ 13 * 13 * 3, 4 });

	torch::Tensor center_anchor_m_ = this->center_anchor_m.clone();
	torch::Tensor corner_anchor_m = util::cxcy_to_xy(center_anchor_m_).view({ 26 * 26 * 3, 4 });

	torch::Tensor center_anchor_s_ = this->center_anchor_s.clone();
	torch::Tensor corner_anchor_s = util::cxcy_to_xy(center_anchor_s_).view({ 52 * 52 * 3, 4 });

	for (int b = 0; b < batch_size; ++b)
	{
		torch::Tensor label = gt_labels[b];
		torch::Tensor corner_gt_box = gt_boxes[b];

		std::vector<int> size({ 13, 26, 52 });

		torch::Tensor center_gt_box = util::xy_to_cxcy(corner_gt_box);

		torch::Tensor scaled_corner_gt_box_l = center_gt_box * float(size[0]);
		torch::Tensor scaled_corner_gt_box_m = center_gt_box * float(size[1]);
		torch::Tensor scaled_corner_gt_box_s = center_gt_box * float(size[2]);

		torch::Tensor iou_anchors_gt_l = util::find_jaccard_overlap(corner_anchor_l, scaled_corner_gt_box_l, 1e-5);
		torch::Tensor iou_anchors_gt_m = util::find_jaccard_overlap(corner_anchor_m, scaled_corner_gt_box_m, 1e-5);
		torch::Tensor iou_anchors_gt_s = util::find_jaccard_overlap(corner_anchor_s, scaled_corner_gt_box_s, 1e-5);

		torch::Tensor scaled_center_gt_box_l = center_gt_box * float(size[0]);
		torch::Tensor scaled_center_gt_box_m = center_gt_box * float(size[1]);
		torch::Tensor scaled_center_gt_box_s = center_gt_box * float(size[2]);

		torch::Tensor bxby_l = scaled_center_gt_box_l.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) });
		torch::Tensor proportion_of_xy_l = bxby_l - bxby_l.floor();
		torch::Tensor bwbh_l = scaled_center_gt_box_l.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) });

		torch::Tensor bxby_m = scaled_center_gt_box_m.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) });
		torch::Tensor proportion_of_xy_m = bxby_m - bxby_m.floor();
		torch::Tensor bwbh_m = scaled_center_gt_box_m.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) });

		torch::Tensor bxby_s = scaled_center_gt_box_s.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) });
		torch::Tensor proportion_of_xy_s = bxby_s - bxby_s.floor();
		torch::Tensor bwbh_s = scaled_center_gt_box_s.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) });

		iou_anchors_gt_l = iou_anchors_gt_l.view({ size[0], size[0], 3, -1 });
		iou_anchors_gt_m = iou_anchors_gt_m.view({ size[1], size[1], 3, -1 });
		iou_anchors_gt_s = iou_anchors_gt_s.view({ size[2], size[2], 3, -1 });

		int num_obj = corner_gt_box.sizes()[0];
		for (int n_obj = 0; n_obj < num_obj; ++n_obj)
		{
			if (IT)
			{
				int cx = bxby_l[n_obj][0].item<int>();
				int cy = bxby_l[n_obj][1].item<int>();
			}
			else
			{
				float l_max = iou_anchors_gt_l.index({ torch::indexing::Ellipsis, n_obj }).max().item<float>();	
				float m_max = iou_anchors_gt_m.index({ torch::indexing::Ellipsis, n_obj }).max().item<float>();
				float s_max = iou_anchors_gt_s.index({ torch::indexing::Ellipsis, n_obj }).max().item<float>();
				std::vector<float> vlms({ l_max, m_max, s_max });
				int best_idx = torch::from_blob(vlms.data(), { (int64_t)vlms.size() }, torch::kFloat32).view({ (int64_t)vlms.size() }).argmax().item<int>();

				if (best_idx == 0)
				{
					int cx = bxby_l[n_obj][0].item<int>();
					int cy = bxby_l[n_obj][1].item<int>();

					std::tuple<torch::Tensor, torch::Tensor> t_max = iou_anchors_gt_l.index({ cy, cx, torch::indexing::Slice(torch::indexing::None), n_obj }).max(0);
					torch::Tensor max_iou = std::get<0>(t_max);
					torch::Tensor max_idx = std::get<1>(t_max);

					int j = max_idx.item<int>();
					gt_objectness_l.index({ b, cy, cx, j, 0 }) = 1;
					gt_prop_txty_l.index({ b, cy, cx, j, torch::indexing::Slice(torch::indexing::None) }) = proportion_of_xy_l[n_obj];

					float anchor_whs_l[] = { this->anchor_whs["large"][2 * j], this->anchor_whs["large"][2 * j + 1] };
					torch::Tensor ratio_of_wh_l = bwbh_l[n_obj] / (torch::from_blob(anchor_whs_l, {2}, torch::kFloat32).to(this->device) / 32);

					gt_twth_l.index({ b, cy, cx, j, torch::indexing::Slice(torch::indexing::None) }) = torch::log(ratio_of_wh_l);
					gt_classes_l.index({ b, cy, cx, j, label[n_obj].item<int>() }) = 1;
				}
				else if (best_idx == 1)
				{
					int cx = bxby_m[n_obj][0].item<int>();
					int cy = bxby_m[n_obj][1].item<int>();

					std::tuple<torch::Tensor, torch::Tensor> t_max = iou_anchors_gt_m.index({ cy, cx, torch::indexing::Slice(torch::indexing::None), n_obj }).max(0);
					torch::Tensor max_iou = std::get<0>(t_max);
					torch::Tensor max_idx = std::get<1>(t_max);

					int j = max_idx.item<int>();
					gt_objectness_m.index({ b, cy, cx, j, 0 }) = 1;
					gt_prop_txty_m.index({ b, cy, cx, j, torch::indexing::Slice(torch::indexing::None) }) = proportion_of_xy_m[n_obj];

					float anchor_whs_m[] = { this->anchor_whs["middle"][2 * j], this->anchor_whs["middle"][2 * j + 1] };
					torch::Tensor ratio_of_wh_m = bwbh_m[n_obj] / (torch::from_blob(anchor_whs_m, { 2 }, torch::kFloat32).to(this->device) / 16);

					gt_twth_m.index({ b, cy, cx, j, torch::indexing::Slice(torch::indexing::None) }) = torch::log(ratio_of_wh_m);
					gt_classes_m.index({ b, cy, cx, j, label[n_obj].item<int>() }) = 1;
				}
				else if (best_idx == 2)
				{
					int cx = bxby_s[n_obj][0].item<int>();
					int cy = bxby_s[n_obj][1].item<int>();

					std::tuple<torch::Tensor, torch::Tensor> t_max = iou_anchors_gt_s.index({ cy, cx, torch::indexing::Slice(torch::indexing::None), n_obj }).max(0);
					torch::Tensor max_iou = std::get<0>(t_max);
					torch::Tensor max_idx = std::get<1>(t_max);

					int j = max_idx.item<int>();
					gt_objectness_s.index({ b, cy, cx, j, 0 }) = 1;
					gt_prop_txty_s.index({ b, cy, cx, j, torch::indexing::Slice(torch::indexing::None) }) = proportion_of_xy_s[n_obj];

					float anchor_whs_s[] = { this->anchor_whs["small"][2 * j], this->anchor_whs["small"][2 * j + 1] };
					torch::Tensor ratio_of_wh_s = bwbh_s[n_obj] / (torch::from_blob(anchor_whs_s, { 2 }, torch::kFloat32).to(this->device) / 8);

					gt_twth_s.index({ b, cy, cx, j, torch::indexing::Slice(torch::indexing::None) }) = torch::log(ratio_of_wh_s);
					gt_classes_s.index({ b, cy, cx, j, label[n_obj].item<int>() }) = 1;
				}

			}
			
			ignore_mask_l[b] = (std::get<0>(iou_anchors_gt_l.max(-1)) < 0.5).toType(torch::kFloat32);
			ignore_mask_m[b] = (std::get<0>(iou_anchors_gt_m.max(-1)) < 0.5).toType(torch::kFloat32);
			ignore_mask_s[b] = (std::get<0>(iou_anchors_gt_s.max(-1)) < 0.5).toType(torch::kFloat32);
		}
	}

	std::vector<torch::Tensor> outputs({
		gt_prop_txty_l, gt_twth_l, gt_objectness_l, gt_classes_l, ignore_mask_l,
		gt_prop_txty_m, gt_twth_m, gt_objectness_m, gt_classes_m, ignore_mask_m,
		gt_prop_txty_s, gt_twth_s, gt_objectness_s, gt_classes_s, ignore_mask_s
		});

	return outputs;
}

std::vector<torch::Tensor> YOLOv4_CoderImpl::split_preds(torch::Tensor pred)
{
	int out_size = pred.sizes()[1];
	pred = pred.view({ -1, out_size, out_size, 3, 5 + this->num_classes });
	torch::Tensor pred_cxcy = pred.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) }).sigmoid();
	torch::Tensor pred_wh = pred.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, 4) });
	torch::Tensor pred_objectness = pred.index({ torch::indexing::Ellipsis, 4 }).unsqueeze(-1).sigmoid();
	torch::Tensor pred_classes = pred.index({ torch::indexing::Ellipsis, torch::indexing::Slice(5, torch::indexing::None) });
	
	return std::vector<torch::Tensor>({ pred_cxcy, pred_wh, pred_objectness, pred_classes });
}

std::vector<torch::Tensor> YOLOv4_CoderImpl::decode(std::vector<torch::Tensor> gcxgcys)
{
	torch::Tensor gcxgcy_l = gcxgcys[0];
	torch::Tensor gcxgcy_m = gcxgcys[1];
	torch::Tensor gcxgcy_s = gcxgcys[2];

	torch::Tensor cxcy_l =
		gcxgcy_l.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) }) +
		this->center_anchor_l.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) }).floor();
	torch::Tensor wh_l =
		torch::exp(gcxgcy_l.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) })) *
		this->center_anchor_l.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) });
	torch::Tensor cxcywh_l = torch::cat({ cxcy_l, wh_l }, -1);

	torch::Tensor cxcy_m =
		gcxgcy_m.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) }) +
		this->center_anchor_m.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) }).floor();
	torch::Tensor wh_m =
		torch::exp(gcxgcy_m.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) })) *
		this->center_anchor_m.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) });
	torch::Tensor cxcywh_m = torch::cat({ cxcy_m, wh_m }, -1);

	torch::Tensor cxcy_s =
		gcxgcy_s.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) }) +
		this->center_anchor_s.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) }).floor();
	torch::Tensor wh_s =
		torch::exp(gcxgcy_s.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) })) *
		this->center_anchor_s.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) });
	torch::Tensor cxcywh_s = torch::cat({ cxcy_s, wh_s }, -1);

	return std::vector<torch::Tensor>({ cxcywh_l, cxcywh_m, cxcywh_s });
}