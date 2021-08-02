import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

fig, ax = plt.subplots()
def draw_bbox(img_path, img_file, target_file_path, x1, y1, x2, y2):
    img_file_path = os.path.join(img_path, img_file)
    if not os.path.exists(img_file_path):
        return
    if os.path.exists(target_file_path):
        return
    try:
        im = np.array(Image.open(img_file_path), dtype=np.uint8)
        draw_bbox_img(im, target_file_path, x1, y1, x2, y2)
    except:
        print('[ERROR] cannot draw bbox', img_file_path)

def draw_bbox_buffer(img_buffer, target_file_path, x1, y1, x2, y2):
    # im = np.array(Image.fromstring(img_buffer), dtype=np.uint8)
    im = np.frombuffer(img_buffer, dtype='uint8')
    draw_bbox_img(im, target_file_path, x1, y1, x2, y2)

def draw_bbox_img(im, target_file_path, x1, y1, x2, y2):
    # # Create figure and axes
    # fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    # Create a Rectangle patch
    # xy, width, height, angle=0.0, **kwargs
    h = y2 - y1
    w = x2 - x1
    rect = patches.Rectangle((x1, y1), w, h, linewidth=5, edgecolor='b', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    fig.savefig(target_file_path) #, dpi=90, bbox_inches='tight')
    # print('save patch', target_file_path)
    plt.cla()

if __name__ == '__main__':
    # img_path = "/data/m1/lim22/multimedia-common-space/Multimedia-Common-Space/aida/data/m9_imgs"
    # img_file = 'IC001FCTM' #"HC0000O5D"
    # patch_path = "/data/m1/lim22/multimedia-common-space/Multimedia-Common-Space/aida/data/m9_patchs"
    img_path = '/data/m1/lim22/multimedia-common-space/Multimedia-Common-Space/aida/voa/data'
    img_file = 'VOA_EN_NW_2012.10.22.1531043_0.jpg'
    patch_path = "/data/m1/lim22/multimedia-common-space/Multimedia-Common-Space/aida/voa/data"
    x1, y1, x2, y2 = 91, 786, 1212, 1203
    target_file_path = os.path.join(patch_path, '%s-%d-%d-%d-%d.png' % (img_file, x1, y1, x2, y2))
    draw_bbox(img_path, img_file, target_file_path, x1, y1, x2, y2)
