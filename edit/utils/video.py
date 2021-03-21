from .path import scandir
import cv2
import os
from .path import get_file_name
from .imageio import imread
import numpy as np


def save_video(video, video_path, factor=1, fps=3, inverse=True):
    '''
        Save a numpy video(rgb) to the disk
    '''
    if isinstance(video, list):
        for i in range(len(video)):
            if isinstance(video[i], np.ndarray):
                assert len(video[i].shape) == 3, 'not video!'
            else:
                raise NotImplementedError("")
        length = len(video)
    elif isinstance(video, np.ndarray):
        assert len(video.shape) == 4, 'not video!'
        length = video.shape[0]

    h, w, _ = video[0].shape

    assert factor >= 1, "factor should >=1"

    if factor > 1:
        if inverse:
            assert w % factor == 0 and h % factor == 0, "w,h should % SR_factor=0"
            for i in range(length):
                video[i] = cv2.resize(video[i], (w//factor, h//factor), interpolation=cv2.INTER_CUBIC)
        else:
            for i in range(length):
                video[i] = cv2.resize(video[i], (int(w*factor), int(h*factor)), interpolation=cv2.INTER_CUBIC)


    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for i in range(length):
        frame = video[i]
        out.write(frame[..., ::-1])
    out.release()


def images2video(dirpath, fps=2, img_suffix='.png', video_suffix='.avi', must_have="idx"):
    """
        give a dir which contains images , trans to video
    """
    imagepathlist = sorted(list(scandir(dirpath, suffix=img_suffix)), key=lambda x:int(x.split("_")[-1][:-4]))
    imagepathlist = [os.path.join(dirpath, v) for v in imagepathlist]
    dn = os.path.dirname(dirpath)
    name = os.path.split(dirpath)[-1]  + "_" + must_have + "_fps-{}".format(fps) + video_suffix

    framelist = []
    id = 0
    for imgpath in imagepathlist:
        if must_have in get_file_name(imgpath):
            id += 1
            framelist.append(imread(imgpath, channel_order='rgb'))
    save_video(framelist, os.path.join(dn, name), fps=fps)
    print("{} ok".format(dirpath))