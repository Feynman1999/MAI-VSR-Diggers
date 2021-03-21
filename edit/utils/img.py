import numpy as np
import random
import cv2
from megengine import Tensor

def img_shelter(img, black_ratio):
    H, W, _ = img.shape
    S = H * (W / 3) / 2
    values = [1, 2, 3, 4]
    a = np.random.choice(values)     # 随机选择顶点
    x = np.random.randint(W / 3, W)  # 随机选择一个边长
    y = 2 * S / x

    scale = random.uniform(0.2, 1)   # 随机选择一个系数
    x = x * scale
    y = y * scale
    
    if a == 1:  # 左上角
        # print('左上角')
        p_A = [0, 0]
        p_B = [x, 0]
        p_C = [0, y]
    elif a == 2:  # 右上角
        # print('右上角')
        p_A = [W, 0]  # 右上角点
        p_B = [W - x, 0]
        p_C = [W, y]
    elif a == 3:  # 左下角
        # print('左下角')
        p_A = [0, H]
        p_B = [0, H - y]
        p_C = [x, H]
    elif a == 4:  # 右下角
        # print('右下角')
        p_A = [W, H]
        p_B = [W - x, H]
        p_C = [W, H - y]

    # img = cv2.imread('C:\\Users\\76397\\Desktop\\stage1\\optical\\1_1.tif', 0)
    # print(img.shape)
    b = np.array([[p_A, p_B, p_C]], dtype=np.int32)  # 左上角点 三个点位置可以随意

    if np.random.random() < black_ratio:
        c = (0, 0, 0) #黑色
    else:
        c = (255, 255, 255) #白色

    cv2.fillPoly(img, b, c) #c是颜色
    # plt.axis('off')
    # plt.imshow(img, cmap='gray')
    # plt.show()
    return img


def bbox_ensemble_back(bbox, Type = 0, Len = 800):
    assert len(bbox.shape) == 2 and bbox.shape[-1] == 4
    bbox_numpy2 = bbox.numpy()
    bbox_numpy = bbox_numpy2.copy()
    bbox_ans = np.zeros_like(bbox_numpy)
    # only change the front 2 now (for hjj)
    if Type == 0:
        return bbox.numpy()
    elif Type == 1:
        bbox_ans = bbox_numpy[:,2] - bbox_numpy[:,0]
        bbox_numpy[:, 2] = Len-1-bbox_numpy[:, 0]
        bbox_numpy[:, 0] = bbox_numpy[:, 2] - bbox_ans
        # 转置
        bbox_ans = np.zeros_like(bbox_numpy)
        bbox_ans[:, 0] = bbox_numpy[:, 1]
        bbox_ans[:, 1] = bbox_numpy[:, 0]
        return bbox_ans
    elif Type == 2:
        # 左右
        bbox_ans = (bbox_numpy[:,3] - bbox_numpy[:,1])
        bbox_numpy[:, 3] = Len-1-bbox_numpy[:, 1]
        bbox_numpy[:, 1] = bbox_numpy[:, 3] - bbox_ans
        # 上下
        bbox_ans = bbox_numpy[:,2] - bbox_numpy[:,0]
        bbox_numpy[:, 2] = Len-1-bbox_numpy[:, 0]
        bbox_numpy[:, 0] = bbox_numpy[:, 2] - bbox_ans
        return bbox_numpy
    elif Type == 3:
        bbox_ans = bbox_numpy[:,3] - bbox_numpy[:,1]
        bbox_numpy[:, 3] = Len-1-bbox_numpy[:, 1]
        bbox_numpy[:, 1] = bbox_numpy[:, 3] - bbox_ans
        # 转置
        bbox_ans = np.zeros_like(bbox_numpy)
        bbox_ans[:, 0] = bbox_numpy[:, 1]
        bbox_ans[:, 1] = bbox_numpy[:, 0]
        return bbox_ans
    elif Type == 4:
        bbox_ans = bbox_numpy[:,3] - bbox_numpy[:,1]
        bbox_numpy[:, 3] = Len-1-bbox_numpy[:, 1]
        bbox_numpy[:, 1] = bbox_numpy[:, 3] - bbox_ans
        return bbox_numpy
    elif Type == 5:
        # 左右
        bbox_ans = (bbox_numpy[:,3] - bbox_numpy[:,1])
        bbox_numpy[:, 3] = Len-1-bbox_numpy[:, 1]
        bbox_numpy[:, 1] = bbox_numpy[:, 3] - bbox_ans
        # 上下
        bbox_ans = bbox_numpy[:,2] - bbox_numpy[:,0]
        bbox_numpy[:, 2] = Len-1-bbox_numpy[:, 0]
        bbox_numpy[:, 0] = bbox_numpy[:, 2] - bbox_ans
        # 转置
        bbox_ans = np.zeros_like(bbox_numpy)
        bbox_ans[:, 0] = bbox_numpy[:, 1]
        bbox_ans[:, 1] = bbox_numpy[:, 0]
        return bbox_ans
    elif Type == 6:
        bbox_ans = bbox_numpy[:,2] - bbox_numpy[:,0]
        bbox_numpy[:, 2] = Len-1-bbox_numpy[:, 0]
        bbox_numpy[:, 0] = bbox_numpy[:, 2] - bbox_ans
        return bbox_numpy
    elif Type == 7:
        bbox_ans[:, 0] = bbox_numpy[:, 1]
        bbox_ans[:, 1] = bbox_numpy[:, 0]
        return bbox_ans
    else:
        raise NotImplementedError("")

def ensemble_for_dim_4(batchdata, Type = 0):
    if Type == 0:
        return batchdata
    elif Type == 1:
        return np.flip(batchdata.transpose(0,1,3,2), axis=2)  # 逆时针旋转90度
    elif Type == 2:
        return np.flip(np.flip(batchdata, axis=2), axis=3)  # 逆时针180
    elif Type == 3:
        return np.flip(batchdata.transpose(0,1,3,2), axis=3)  # 逆时针270
    elif Type == 4:
        return np.flip(batchdata, axis = 3)  # 原图左右翻转
    elif Type == 5:
        return np.flip(np.flip(batchdata.transpose(0,1,3,2), axis=2), axis= 3)  # 
    elif Type == 6:
        return np.flip(batchdata, axis = 2)  # 原图上下翻转
    elif Type == 7:
        return batchdata.transpose(0,1,3,2)  # 原图转置
    else:
        raise NotImplementedError("")

def ensemble_forward(batchdata, Type = 0):
    """
        batchdata:  B,C,H,W    numpy     or  B,T,C,H,W
        Type:  0~7
        return: ndarray
    """
    assert is_ndarray(batchdata)
    assert len(batchdata.shape) == 4 or len(batchdata.shape) == 5
    if len(batchdata.shape) == 4:
        return ensemble_for_dim_4(batchdata, Type)
    else:
        L = []
        for i in range(batchdata.shape[1]):
            L.append(ensemble_for_dim_4(batchdata[:, i, ...], Type=Type))
        return np.stack(L, axis = 1)  # B,T,C,H,W


def ensemble_back_for_dim_4(batchdata, Type):
    if Type == 0:
        return batchdata
    elif Type == 1:
        return np.flip(batchdata.transpose(0,1,3,2), axis=3)
    elif Type == 2:
        return np.flip(np.flip(batchdata, axis=2), axis=3)
    elif Type == 3:
        return np.flip(batchdata.transpose(0,1,3,2), axis=2) 
    elif Type == 4:
        return np.flip(batchdata, axis = 3)
    elif Type == 5:
        return np.flip(np.flip(batchdata, axis = 3).transpose(0,1,3,2), axis=3)
    elif Type == 6:
        return np.flip(batchdata, axis = 2)
    elif Type == 7:
        return batchdata.transpose(0,1,3,2)
    else:
        raise NotImplementedError("")

def ensemble_back(batchdata, Type = 0):
    """
        batchdata:  B,C,H,W    tensor
        Type: 0~7
        return: ndarray
    """
    assert is_var(batchdata) or is_ndarray(batchdata)
    if is_var(batchdata):
        batchdata = batchdata.to("cpu0").astype('float32').numpy()
    if len(batchdata.shape) == 4:
        return ensemble_back_for_dim_4(batchdata, Type)
    elif len(batchdata.shape) == 5:
        # raise RuntimeError("ensemble back should for 4 dim not 5 dim usually!")
        L = []
        for i in range(batchdata.shape[1]):
            L.append(ensemble_back_for_dim_4(batchdata[:, i, ...], Type=Type))
        return np.stack(L, axis = 1)  # B,T,C,H,W

def _half(x):
    if x % 2 ==0:
        return (x//2, x//2)
    else:
        return (x//2, x//2 +1)

def cal_pad_size(l, padding_multi):
    diff = l % padding_multi
    if diff == 0:
        return (0, 0)
    diff = padding_multi - diff
    return _half(diff)

def img_multi_padding(img, padding_multi = 4, pad_value = 0, pad_method = "reflect"):
    """
        pad_method: reflect  edge  constant    
    """
    if padding_multi <= 1:
        return img
    assert isinstance(img, np.ndarray)
    dimlen = len(img.shape)
    origin_H = img.shape[-2]
    origin_W = img.shape[-1]
    pad_H = cal_pad_size(origin_H, padding_multi)
    pad_W = cal_pad_size(origin_W, padding_multi)
    if dimlen == 5: # [B,N,C,H,W]
        return np.pad(img, ((0,0), (0,0), (0,0), pad_H, pad_W), mode=pad_method)
    elif dimlen == 4:
        return np.pad(img, ((0,0), (0,0), pad_H, pad_W), mode=pad_method)
    elif dimlen == 3:
        return np.pad(img, ((0,0), pad_H, pad_W), mode=pad_method)
    elif dimlen == 2:
        return np.pad(img, (pad_H, pad_W), mode=pad_method)
    else:
        raise NotImplementedError("dimlen: {} not implement".format(dimlen))

def img_de_multi_padding(img, origin_H, origin_W):
    """
        support tensor
    """
    dimlen = len(img.shape)
    paded_H = img.shape[-2]
    paded_W = img.shape[-1]
    pad_H = _half(paded_H - origin_H)
    pad_W = _half(paded_W - origin_W)
    if dimlen in (2, 3, 4, 5):
        return img[..., pad_H[0]: paded_H - pad_H[1], pad_W[0]: paded_W - pad_W[1]]
    else:
        raise NotImplementedError("dimlen: {} not implement".format(dimlen))

def _scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)


interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def imresize(img,
             size,
             return_scale=False,
             interpolation='area',
             out=None):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
        out (ndarray): The output destination.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    assert interpolation in interp_codes
    resized_img = cv2.resize(img, size, dst=out, interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def imresize_like(img, dst_img, return_scale=False, interpolation='area'):
    """Resize image to the same size of a given image.

    Args:
        img (ndarray): The input image.
        dst_img (ndarray): The target image.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Same as :func:`resize`.

    Returns:
        tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = dst_img.shape[:2]
    return imresize(img, (w, h), return_scale, interpolation)


def rescale_size(old_size, scale, return_scale=False):
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(img, scale, return_scale=False, interpolation='area'):
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def bboxflip_(bbox, direction='horizontal', Len = 400):
    """
        top left  bottom  right
    """
    assert direction in ['horizontal', 'vertical']
    assert len(bbox.shape) == 1 and len(bbox) == 4
    if direction == 'horizontal':
        x = bbox[3] - bbox[1]
        bbox[3] = Len-1-bbox[1]
        bbox[1] = bbox[3] - x 
    else:
        x = bbox[2] - bbox[0]
        bbox[2] = Len-1-bbox[0]
        bbox[0] = bbox[2] - x

def imflip_(img, direction='horizontal'):
    """Inplace flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or "vertical".

    Returns:
        ndarray: The flipped image (inplace).
    """
    assert direction in ['horizontal', 'vertical']
    if direction == 'horizontal':
        return cv2.flip(img, 1, img)
    else:
        return cv2.flip(img, 0, img)

def flowflip_(flow, direction):
    imflip_(flow, direction)
    # 变负号
    if direction == 'horizontal':
        flow[:,:,1] *= (-1)
    else:
        flow[:,:,0] *= (-1)

def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             border_value=0,
             auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=border_value)
    return rotated


def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)


def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img


def is_var(tensor):
    return isinstance(tensor, Tensor)


def is_ndarray(tensor):
    return isinstance(tensor, np.ndarray)


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert variable or ndarray into image numpy arrays.

    After clamping to (min, max), image values will be normalized to [0, 1].

    For differnet tensor shapes, this function will have different behaviors:

        1. 4D mini-batch Tensor of shape (N x 3/1 x H x W):
            Use `make_grid` to stitch images in the batch dimension, and then
            convert it to numpy array.
        2. 3D Tensor of shape (3/1 x H x W) and 2D Tensor of shape (H x W):
            Directly change to numpy array.

    Note that the image channel in input tensors should be RGB order. This
    function will convert it to cv2 convention, i.e., (H x W x C) with BGR
    order.

    Args:
        tensor (Tensor | list[Tensor]): Input tensors.
        out_type (numpy type): Output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple): min and max values for clamp.

    Returns:
        (Tensor | list[Tensor]): 3D ndarray of shape (H x W x C) or 2D ndarray
        of shape (H x W).
    """
    if is_var(tensor):
        tensor = tensor.to("cpu0").astype('float32').numpy()
    elif isinstance(tensor, list) and all(is_var(t) for t in tensor):
        tensor = [t.to("cpu0").astype('float32').numpy() for t in tensor]
    else:
        assert is_ndarray(tensor) or (isinstance(tensor, list) and all(is_ndarray(t) for t in tensor))

    if is_ndarray(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        # Squeeze two times so that:
        # 1. (1, 1, h, w) -> (h, w) or
        # 3. (1, 3, h, w) -> (3, h, w) or
        # 2. (n>1, 3/1, h, w) -> (n>1, 3/1, h, w)
        _tensor = np.squeeze(_tensor)
        _tensor = np.clip(_tensor, a_min=min_max[0], a_max=min_max[1])
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = len(_tensor.shape)
        if n_dim == 4:
            raise NotImplementedError("dose not support mini batch var2image")
            # img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 3:
            img_np = np.transpose(_tensor[[2, 1, 0], :, :], (1, 2, 0))  # CHW -> HWC and rgb -> bgr
        elif n_dim == 2:
            img_np = _tensor
        else:
            raise ValueError('Only support 4D, 3D or 2D tensor. '
                             f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result
