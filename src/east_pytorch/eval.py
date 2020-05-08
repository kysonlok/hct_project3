import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import EAST
from dataset import get_rotate_mat
from shapely.geometry import Polygon

#图片读入并缩放
def resize_img(img):
	'''resize image to be divisible by 32
	'''
	w, h = img.size
	resize_w = w
	resize_h = h

	resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
	resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
	img = img.resize((resize_w, resize_h), Image.BILINEAR)
	ratio_h = resize_h / h
	ratio_w = resize_w / w

	return img, ratio_h, ratio_w


def load_pil(img):
	'''convert PIL Image to torch.Tensor
	'''
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
	return t(img).unsqueeze(0)

#多边形是否合法
def is_valid_poly(res, score_shape, scale):
	'''check if the poly in image scope
	Input:
		res        : restored poly in original image
		score_shape: score map shape
		scale      : feature map -> image
	Output:
		True if valid
	'''
	cnt = 0
	for i in range(res.shape[1]):
		if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False

# 把d1,d2,d3,d4转换为四变行的8个顶点
def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
	'''restore polys from feature maps in given positions
	Input:
		valid_pos  : potential text positions <numpy.ndarray, (n,2)>
		valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
		score_shape: shape of score map
		scale      : image / feature map
	Output:
		restored polys <numpy.ndarray, (n,8)>, index
	'''
	polys = []
	index = []
	valid_pos *= scale
	d = valid_geo[:4, :] # 4 x N
	angle = valid_geo[4, :] # N,

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		# d1, d2, d3, d4
		y_min = y - d[0, i]
		x_max = x + d[1, i]
		y_max = y + d[2, i]
		x_min = x - d[3, i]
		rotate_mat = get_rotate_mat(-angle[i])
		
		temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
		temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
		coordidates = np.concatenate((temp_x, temp_y), axis=0)
		res = np.dot(rotate_mat, coordidates)
		res[0,:] += x
		res[1,:] += y
		
		if is_valid_poly(res, score_shape, scale):
			index.append(i)
			polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
	return np.array(polys), index

def intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g


def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    return S[keep]

def  locality_aware_nms(boxes, nms_thresh):
    #[这里实现locality_aware_nms]
    '''实现locallity_aware_nms
    Input:
            boxes         :polys,quad的形式的polys,poly是四边形得四个顶点的坐标（x1,y1,x2,y2,x3,y3,x4,y4)<numpy.ndarray,(n,9)>
            nms_thresh    :threshold in nms,<float>
    Output:
            boxes         :final polys <numpy.ndarray,(n,9)>
    '''
    S = []
    p = None
    for g in boxes:
        if p is not None and intersection(g, p) > nms_thresh:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), nms_thresh)

def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
	'''get boxes from feature map
	Input:
		score       : score map from model <numpy.ndarray, (1,row,col)>
		geo         : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	# 整个图片上得点，每个点代表一个候选框
	score = score[0,:,:]
	#留下score大于score_thresh的候选框对应的像素点坐标
	#方便理解，抄下此句：xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	xy_text = np.argwhere(score > score_thresh) # shape: (n, 2)

	if xy_text.size == 0:
		return None
	#按行排序
	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	#得到像素点坐标
	# 将二维数组转成坐标，eg: 二维数组(0, 2)的元素，对应的坐标是(2, 0)
	valid_pos = xy_text[:, ::-1].copy() # shape: (n, 2)
	# 利用置信度大于阈值的元素，取候选框的geo值
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # shape: (5, n)
	#把d1,d2,d3,d4转换为四个顶点的坐标
	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape) 
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
	#[这里自己实现] locality-aware nms:
	#nms_thresh是float
	#boxes是quad的形式，四边形得四个顶点的坐标（x1,y1,x2,y2,x3,y3,x4,y4)
	boxes = locality_aware_nms(boxes.astype('float32'), nms_thresh)
	return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
	'''refine boxes
	Input:
		boxes  : detected polys <numpy.ndarray, (n,9)>
		ratio_w: ratio of width
		ratio_h: ratio of height
	Output:
		refined boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	return np.around(boxes)
	
	
def detect(img, model, device):
	'''detect text regions of img using model
	Input:
		img   : PIL Image
		model : detection model
		device: gpu if gpu is available
	Output:
		detected polys
	'''
	# 图片缩放
	img, ratio_h, ratio_w = resize_img(img)
	# 计算出score_map,geo_map
	with torch.no_grad():
		score, geo = model(load_pil(img).to(device))
	#get_boxes函数里面用到 locality aware NMS
	boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
	return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
	'''plot boxes on image
	'''
	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	for box in boxes:
	#绘制出四边形
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
	return img


def detect_dataset(model, device, test_img_path, submit_path):
	'''detection on whole dataset, save .txt results in submit_path
	Input:
		model        : detection model
		device       : gpu if gpu is available
		test_img_path: dataset path
		submit_path  : submit result for evaluation
	'''
	img_files = os.listdir(test_img_path)
	img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])
	
	for i, img_file in enumerate(img_files):
			#print('evaluating {} image'.format(i), end='\r')
			boxes = detect(Image.open(img_file), model, device)
			seq = []
			if boxes is not None:
				seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes])
			with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 'w') as f:
				f.writelines(seq)

def main():
    img_path = os.path.abspath('./ICDAR_2015/test_img/img_2.jpg')
    model_path = './pths/model_epoch_300_txy.pth'
    res_img = 'res.bmp'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EAST()
    model = model.eval()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    img = Image.open(img_path)

    boxes = detect(img, model, device)
	#绘制boxes到图片上
    plot_img = plot_boxes(img, boxes)	
	#保存图片到./res.bmp
    plot_img.save(res_img)

if __name__ == "__main__":
    main()