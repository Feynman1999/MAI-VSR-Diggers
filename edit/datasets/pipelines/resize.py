import random
from megengine.data.transform import RandomResizedCrop as mge_RRC
from megengine.data.transform import Resize as mge_resize
from ..registry import PIPELINES
from edit.utils import interp_codes

@PIPELINES.register_module()
class Resize(object):
    """
        Args:
        size (int|list|tuple): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int): Interpolation mode of resize. Default: cv2.INTER_LINEAR.
    """
    def __init__(self, keys, size, interpolation='bilinear'):
        assert interpolation in interp_codes
        self.keys = keys
        self.size = size
        self.interpolation_str = interpolation
        self.resize = mge_resize(output_size=self.size, interpolation=interp_codes[interpolation])

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    self.resize.apply(v) for v in results[key]
                ]
            else:
                results[key] = self.resize.apply(results[key])
        return results

    def __repr__(self):
        interpolate_str = self.interpolation_str
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

@PIPELINES.register_module()
class RandomResizedCrop(object):
    """
        Crop the input data to random size and aspect ratio.
        A crop of random size (default: of 0.08 to 1.0) of the original size and a random
        aspect ratio (default: of 3/4 to 1.33) of the original aspect ratio is made.
        After applying crop transfrom, the input data will be resized to given size.

        Args:
            output_size (int|list|tuple): Target size of output image, with (height, width) shape.
            scale (list|tuple): Range of size of the origin size cropped. Default: (0.08, 1.0)
            ratio (list|tuple): Range of aspect ratio of the origin aspect ratio cropped. Default: (0.75, 1.33)
            interpolation: 
                'nearest': cv2.INTER_NEAREST,
                'bilinear': cv2.INTER_LINEAR,
                'bicubic': cv2.INTER_CUBIC,
                'area': cv2.INTER_AREA,
                'lanczos': cv2.INTER_LANCZOS4
    """
    def __init__(self, keys, output_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear', do_prob = 0.5):
        assert interpolation in interp_codes
        self.keys = keys
        self.size = output_size
        self.interpolation_str = interpolation
        self.scale = scale
        self.ratio = ratio
        self.rrc = mge_RRC(output_size=output_size, scale_range=scale, ratio_range=ratio, interpolation=interp_codes[interpolation])
        self.do_prob = do_prob

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if random.random() < self.do_prob:
            for key in self.keys:
                if isinstance(results[key], list):
                    results[key] = [
                        self.rrc.apply(v) for v in results[key]
                    ]
                else:
                    results[key] = self.rrc.apply(results[key])
            return results
        else:
            return results
        
    def __repr__(self):
        interpolate_str = self.interpolation_str
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

