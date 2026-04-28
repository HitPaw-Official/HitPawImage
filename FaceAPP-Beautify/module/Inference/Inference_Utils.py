import numpy as np
from math import ceil
from itertools import product
import onnxruntime 

def init_model(modelname, device, deviceid=0):
    # if device == "CoreML":
    #     import coremltools as ct
    #     session = ct.models.MLModel(modelname, True)
    # else:
    #     # <= 5仅供FaceRestore 模型使用
    #     if len(modelname) <= 5:
    #         modelrootpath = './model/onnx/HP_FACE_ENHANCEMENT'
    #         model_path = modelrootpath + "/" + modelname + '.onnx'
    #         # session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider']) CUDAExecutionProvider
    #         session = rt.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    #     else:
    #         # session = onnxruntime.InferenceSession(modelname, providers=['DmlExecutionProvider','CPUExecutionProvider'])
    #         session = rt.InferenceSession(modelname, providers=['CUDAExecutionProvider'])
    # dml_option = {'device_id': 0}
    # device_dml = [('DmlExecutionProvider', dml_option)]
    device_dml = ['DmlExecutionProvider', 'CPUExecutionProvider']
    sess_options = onnxruntime.SessionOptions()
    sess_options.enable_mem_pattern = False
    session = onnxruntime.InferenceSession(modelname, sess_options, providers=device_dml) #providers=['DmlExecutionProvider'])
    return session


### 人脸修复

def normalize_image(image, normalize_type='255'):
    """
    Normalize image

    Parameters
    ----------
    image: numpy array
        The image you want to normalize
    normalize_type: string
        Normalize type should be chosen from the type below.
        - '255': simply dividing by 255.0
        - '127.5': output range : -1 and 1
        - 'ImageNet': normalize by mean and std of ImageNet
        - 'None': no normalization

    Returns
    -------
    normalized_image: numpy array
    """
    if normalize_type == 'None':
        return image
    elif normalize_type == '255':
        return image / 255.0
    elif normalize_type == '127.5':
        return image / 127.5 - 1.0
    elif normalize_type == 'ImageNet':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image / 255.0
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        return image
    else:
        print(f'Unknown normalize_type is given: {normalize_type}')

def name_in(sess, id=0):
    return sess.get_inputs()[id].name

def name_out(sess, id=0):
    return sess.get_outputs()[id].name

def leaky_relu(data, negative=0.2):
    dmax = np.clip(data, 0., 100)
    dmin = np.clip(data, -100., 0.)
    dst = dmax + negative * dmin
    return dst

### 人脸检测

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[(inds + 1)]

    return keep

def cal_ratio(w0, h0, w, h):
    ratio = 1
    r_w = w / w0
    r_h = h / h0
    ratio = r_w if r_w < r_h else r_h
    return ratio

def onnx_decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = np.concatenate((
     priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
     priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])),
      axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def onnx_decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
     priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
     priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
     priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
     priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]),
      axis=1)
    return landms

class OnnxPriorBox(object):

    def __init__(self, cfg, image_size=None, phase='train'):
        super(OnnxPriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]
        self.name = 's'

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output = np.clip(output, 0, 1)
        return output
