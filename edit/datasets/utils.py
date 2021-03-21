import os

def get_key_for_video_imgs(x):
    """
        x: "calendar/0000.png"
    """
    clip, name = x.split("/")
    name, _ = os.path.splitext(name)
    return (clip, int(name))  # 按照clip升序sort，若属于同一clip，则按照frame number升序
