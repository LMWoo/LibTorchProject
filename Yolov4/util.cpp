#include "util.h"

torch::Tensor util::cxcy_to_xy(torch::Tensor cxcy)
{
	torch::Tensor x1y1 =
		cxcy.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) }) -
		(cxcy.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) }) / 2);
	torch::Tensor x2y2 =
		cxcy.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) }) +
		(cxcy.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) }) / 2);
	return torch::cat({ x1y1, x2y2 }, -1);
}

torch::Tensor util::xy_to_cxcy(torch::Tensor xy)
{
	torch::Tensor cxcy = (
		xy.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) }) +
		xy.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) })
		) / 2;
	torch::Tensor wh =
		xy.index({ torch::indexing::Ellipsis, torch::indexing::Slice(2, torch::indexing::None) }) -
		xy.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 2) });
	return torch::cat({ cxcy, wh }, -1);
}

torch::Tensor util::find_jaccard_overlap(torch::Tensor set_1, torch::Tensor set_2, float eps)
{
	torch::Tensor intersection = find_intersection(set_1, set_2);

	torch::Tensor areas_set_1 =
		(set_1.index({ torch::indexing::Slice(torch::indexing::None), 2 }) - set_1.index({ torch::indexing::Slice(torch::indexing::None), 0 })) *
		(set_1.index({ torch::indexing::Slice(torch::indexing::None), 3 }) - set_1.index({ torch::indexing::Slice(torch::indexing::None), 1 }));
	torch::Tensor areas_set_2 =
		(set_2.index({ torch::indexing::Slice(torch::indexing::None), 2 }) - set_2.index({ torch::indexing::Slice(torch::indexing::None), 0 })) *
		(set_2.index({ torch::indexing::Slice(torch::indexing::None), 3 }) - set_2.index({ torch::indexing::Slice(torch::indexing::None), 1 }));

	torch::Tensor uni = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection + eps;

	return intersection / uni;
}

torch::Tensor util::find_intersection(torch::Tensor set_1, torch::Tensor set_2)
{
	torch::Tensor lower_bounds = torch::max(
		set_1.index({ torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice({torch::indexing::None, 2}) }).unsqueeze(1),
		set_2.index({ torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice({torch::indexing::None, 2}) }).unsqueeze(0)
	);
	torch::Tensor upper_bounds = torch::min(
		set_1.index({ torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice({2, torch::indexing::None}) }).unsqueeze(1),
		set_2.index({ torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice({2, torch::indexing::None}) }).unsqueeze(0)
	);

	torch::Tensor intersection_dims = torch::clamp(upper_bounds - lower_bounds, 0);

	torch::Tensor output =
		intersection_dims.index({ torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None), 0 }) *
		intersection_dims.index({ torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None), 1 });

	return output;
}