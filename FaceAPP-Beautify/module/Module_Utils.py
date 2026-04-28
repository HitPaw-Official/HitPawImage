import numpy as np
from PIL import Image, ImageFilter

REFERENCE_FACIAL_POINTS = [
 [
  30.29459953, 51.69630051],
 [
  65.53179932, 51.50139999],
 [
  48.02519989, 71.73660278],
 [
  33.54930115, 92.3655014],
 [
  62.72990036, 92.20410156]]

face_mask = None

DEFAULT_CROP_SIZE = (96, 112)

def arr_to_img(arr):
    arr = arr[0].transpose(1, 2, 0) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img


def img_to_arr(img):
    arr = np.asarray(img)
    arr = arr.astype(np.uint8) / 255.0
    img = np.array([arr]).astype(np.float32).transpose((0, 3, 1, 2))
    return img

def get_reference_facial_points(output_size=None, inner_padding_factor=0.0, outer_padding=(0, 0), default_square=False):
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff
    else:
        if output_size:
            if output_size[0] == tmp_crop_size[0]:
                if output_size[1] == tmp_crop_size[1]:
                    print('output_size == DEFAULT_CROP_SIZE {}: return default reference points'.format(tmp_crop_size))
                    return tmp_5pts
        if inner_padding_factor == 0 and outer_padding == (0, 0):
            if output_size is None:
                print('No paddings to do: return default reference points')
                return tmp_5pts
            raise FaceWarpException('No paddings to do, output_size must be None or {}'.format(tmp_crop_size))
    if not 0 <= inner_padding_factor <= 1.0:
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')
    if inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0:
        if output_size is None:
            output_size = tmp_crop_size * (1 + inner_padding_factor * 2).astype(np.int32)
            output_size += np.array(outer_padding)
            print('              deduced from paddings, output_size = ', output_size)
    if not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1]):
        raise FaceWarpException('Not (outer_padding[0] < output_size[0]and outer_padding[1] < output_size[1])')
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise FaceWarpException('Must have (output_size - outer_padding)= some_scale * (crop_size * (1.0 + inner_padding_factor)')
    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    tmp_5pts = tmp_5pts * scale_factor
    tmp_crop_size = size_bf_outer_pad
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size
    return reference_5point


def _umeyama(src, dst, estimate_scale=True, scale=1.0):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=(np.double))
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye((dim + 1), dtype=(np.double))
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[(dim - 1)]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = scale
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return (
     T, scale)

def warp_and_crop_face(src_img, facial_pts, reference_pts=None, crop_size=(96, 112), align_type='smilarity'):
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = (0, 0)
            output_size = crop_size
            reference_pts = get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)
    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or (min(ref_pts_shp) != 2):
        raise FaceWarpException('reference_pts.shape must be (K,2) or (2,K) and K>2')
    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T
    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or (min(src_pts_shp) != 2):
        raise FaceWarpException('facial_pts.shape must be (K,2) or (2,K) and K>2')
    if src_pts_shp[0] == 2:
        src_pts = src_pts.T
    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException('facial_pts and reference_pts must have the same shape')
    params, scale = _umeyama(src_pts, ref_pts)
    tfm = params[:2, :]
    params, _ = _umeyama(ref_pts, src_pts, False, scale=(1.0 / scale))
    tfm_inv = params[:2, :]
    tfm_data = tfm_inv.reshape(-1)[:6]
    face_img = src_img.transform((crop_size[0], crop_size[1]), method=(Image.AFFINE), data=tfm_data, resample=(Image.BILINEAR))
    face_arr = np.asarray(face_img)

    return (face_arr, tfm)

def init_face_mask():
    global face_mask
    if face_mask is None:
        print('init face mask array')
        face_mask = np.zeros((512, 512), np.float32)
        face_mask[26:487, 26:487] = 1
        mask_img = Image.fromarray(((face_mask * 255).astype(np.uint8)), mode='L')
        blur_mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=11))
        blur_mask_img = blur_mask_img.filter(ImageFilter.GaussianBlur(radius=11))
        face_mask = np.asarray(blur_mask_img)
    return face_mask

def parsmask(out):
    # out = out.argmax(dim=1).squeeze().cpu().numpy()
    mask = np.zeros(out.shape)
    # MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
    # for idx, color in enumerate(MASK_COLORMAP):
    #     mask[out == idx] = color
    mask = np.where(out>0,1,0)
    # mask[:26, :26] = 1
    # mask[487:, 487:] = 1
    # mask[:26, 487:] = 1
    # mask[487:, :26] = 1
    thres = 30
    mask[:thres, :] = 0
    mask[-thres:, :] = 0
    mask[:, :thres] = 0
    mask[:, -thres:] = 0
    #  blur the mask
    mask_img = Image.fromarray(((mask*255).astype(np.uint8)), mode='L')
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=7))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=7))
    # mask = cv2.GaussianBlur(mask, (101, 101), 11)
    # mask = cv2.GaussianBlur(mask, (101, 101), 11)
    # remove the black borders

    mask = np.array(mask_img)
    # mask = mask / 255.
    return mask

class FaceWarpException(Exception):
    pass