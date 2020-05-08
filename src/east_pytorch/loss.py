import torch
import torch.nn as nn


def get_class_balanced_cross_entropy(gt_score,pred_score):
    #请写出类平衡交叉熵的liss
	a = gt_score.shape[2]
	b = gt_score.shape[3]
	beta = 1-torch.sum(gt_score)/(a*b)
 
	score_loss = -beta * gt_score * torch.log(pred_score+1e-5) - (1-beta) * (1-gt_score )* torch.log(1-pred_score+1e-5)
	# print(torch.mean(score_loss))

	return torch.sum(score_loss) / torch.sum(gt_score)

def dice_coefficient(gt_score, pred_score):
	smooth = 1e-5
	
	intersection = torch.sum(gt_score * pred_score)
	union = torch.sum(gt_score) + torch.sum(pred_score)
	
	loss = 1. - (2. * (intersection + smooth)) / (union + smooth)

	return loss

def get_geo_loss(gt_geo, pred_geo):
    #写出d1,d2,d3,d4,4个feature map的iou_loss 和 angle_map的loss
	# d1 -> top, d2 -> right, d3 -> bottom, d4 -> left
	d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(gt_geo, 1, 1)
	d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(pred_geo, 1, 1)
	
	area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
	area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
	w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
	h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
	area_intersect = w_union * h_union
	area_union = area_gt + area_pred - area_intersect
	
	iou_loss = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
	angle_loss = 1 - torch.cos(theta_pred - theta_gt)

	return iou_loss, angle_loss

class Loss(nn.Module):
	def __init__(self, weight_angle=10):
		super(Loss, self).__init__()
		self.weight_angle = weight_angle

	def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
		# 过滤没有文字目标
		if torch.sum(gt_score) < 1:
			return torch.sum(pred_score + pred_geo) * 0
		
		# classify_loss = get_class_balanced_cross_entropy(gt_score, pred_score*(1-ignored_map))
		# 计算 score loss 使用 dice loss 代替分类平衡交叉熵 loss 
		classify_loss = dice_coefficient(gt_score, pred_score*(1-ignored_map))

		# IoU loss + Angle loss
		iou_loss, angle_loss = get_geo_loss(gt_geo, pred_geo)

		iou_loss = torch.sum(iou_loss * gt_score) / torch.sum(gt_score)
		angle_loss = torch.sum(angle_loss * gt_score) / torch.sum(gt_score)

		geo_loss = self.weight_angle * angle_loss + iou_loss

		# print('classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss, angle_loss, iou_loss))
		
		return geo_loss + classify_loss
