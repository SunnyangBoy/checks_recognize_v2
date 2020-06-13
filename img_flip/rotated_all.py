import os
import torch
import numpy as np
from PIL import Image
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

createImg_root = '/mnt/valid_rotate180/images'
createLab_root = '/mnt/valid_rotate180/labels'

def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def rotate_img(img_dir, img_savepath, file_path, lab_savepath, flag):
    image = Image.open(img_dir)
    if flag:
        image = image.rotate(180, Image.BILINEAR)
    image.save(img_savepath)

    with open(lab_savepath, 'w') as writer:
        with open(file_path, 'r') as lines:
            lines = lines.readlines()
            for l, line in enumerate(lines):
                line = line.split(';')
                vertice = [int(vt) for vt in line[1:-1]]
                vertice = np.array(vertice)
                if flag:
                    center_x = (image.width - 1) / 2
                    center_y = (image.height - 1) / 2
                    new_vertice = np.zeros(vertice.shape)
                    new_vertice[:] = rotate_vertices(vertice, - math.pi, np.array([[center_x], [center_y]]))
                    vertice = new_vertice
                new_line = []
                new_line.append(line[0])
                for v in vertice:
                    new_line.append(str(int(v)))
                new_line.append(line[-1])
                new_line = ';'.join(new_line)
                writer.write(new_line)
        writer.close()


if __name__ == '__main__':

    img_label_dir = '/mnt/valid_rotate90/labels'
    for root, dirs, files in os.walk(img_label_dir):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            image_name = file[0: -4] + '.jpg'
            image_dir = os.path.join('/mnt/valid_rotate90/images', image_name)
            with open(file_path, 'r') as lines:
                lines = lines.readlines()
                first_y = int(lines[0].split(';')[2])
                last_y = int(lines[0].split(';')[-2])
                flag = True
                if first_y < last_y:
                    flag = False
                img_savepath = os.path.join(createImg_root, image_name)
                lab_savepath = os.path.join(createLab_root, file)
                rotate_img(image_dir, img_savepath, file_path, lab_savepath, flag)
