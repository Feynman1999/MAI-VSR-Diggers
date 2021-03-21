import numpy as np
import cv2

def adjust_brightness(img, factor=1.):
    """Adjust image brightness.
    This function controls the brightness of an image. An
    enhancement factor of 0.0 gives a black image.
    A factor of 1.0 gives the original image. This function
    blends the source image and the degenerated black image:
    ``output = img * factor + degenerated * (1 - factor)``
    Args:
        img (ndarray): Image to be brightened.
        factor (float): A value controls the enhancement.
            Factor 1.0 returns the original image, lower
            factors mean less color (brightness, contrast,
            etc), and higher values more. Default 1.
    Returns:
        ndarray: The brightened image.
    """
    degenerated = np.zeros_like(img)
    # Note manually convert the dtype to np.float32, to
    # achieve as close results as PIL.ImageEnhance.Brightness.
    # Set beta=1-factor, and gamma=0
    brightened_img = cv2.addWeighted(
        img.astype(np.float32), factor, degenerated.astype(np.float32),
        1 - factor, 0)
    return brightened_img.astype(img.dtype)


def adjust_contrast(img, factor=1.):
    """Adjust image contrast.
    This function controls the contrast of an image. An
    enhancement factor of 0.0 gives a solid grey
    image. A factor of 1.0 gives the original image. It
    blends the source image and the degenerated mean image:
    ``output = img * factor + degenerated * (1 - factor)``
    Args:
        img (ndarray): Image to be contrasted. BGR order.
        factor (float): Same as :func:`mmcv.adjust_brightness`.
    Returns:
        ndarray: The contrasted image.
    """
    gray_img = bgr2gray(img)
    hist = np.histogram(gray_img, 256, (0, 255))[0]
    mean = round(np.sum(gray_img) / np.sum(hist))
    degenerated = (np.ones_like(img[..., 0]) * mean).astype(img.dtype)
    degenerated = gray2bgr(degenerated)
    contrasted_img = cv2.addWeighted(
        img.astype(np.float32), factor, degenerated.astype(np.float32),
        1 - factor, 0)
    return contrasted_img.astype(img.dtype)