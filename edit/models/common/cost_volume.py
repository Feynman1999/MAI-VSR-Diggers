import megengine as mge
import megengine.functional as F
import collections

def add_H_W_Padding(x, margin):
    shape = x.shape
    padding_shape = list(shape)[:-2] + [ shape[-2] + 2*margin, shape[-1] + 2*margin ]
    res = mge.zeros(padding_shape, dtype=x.dtype)
    res = res.set_subtensor(x)[:, :, margin:margin + shape[-2],  margin: margin + shape[-1]]
    return res

def compute_cost_volume(features1, features2, max_displacement):
    """Compute the cost volume between features1 and features2.
    Displace features2 up to max_displacement in any direction and compute the
    per pixel cost of features1 and the displaced features2.
    Args:
        features1: tensor of shape [b, c, h, w]
        features2: tensor of shape [b, c, h, w]
        max_displacement: int, maximum displacement for cost volume computation.
    Returns:
        tensor of shape [b, (2 * max_displacement + 1) ** 2, h, w] of costs for
        all displacements.
    """
    # Set maximum displacement and compute the number of image shifts.
    _, _, height, width = features1.shape
    # if max_displacement <= 0 or max_displacement >= height:
    #     raise ValueError(f'Max displacement of {max_displacement} is too large.')

    max_disp = max_displacement
    num_shifts = 2 * max_disp + 1

    # Pad features2 and shift it while keeping features1 fixed to compute the cost volume through correlation.
    # Pad features2 such that shifts do not go out of bounds.
    features2_padded = add_H_W_Padding(features2, margin = max_disp)
    cost_list = []
    for i in range(num_shifts):
        for j in range(num_shifts):
            corr = F.mean(
                features1 * features2_padded[:, :, i:(height+i), j:(width + j)],
                axis = 1,
                keepdims = True
            ) # [B, 1, H, W]
            cost_list.append(corr)
    cost_volume = F.concat(cost_list, axis = 1)
    return cost_volume, features2_padded
