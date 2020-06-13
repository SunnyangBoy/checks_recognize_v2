import os
import torch
from model import DetectAngleModel
import numpy as np
from PIL import Image
import math
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    model = DetectAngleModel()
    model.load_state_dict(torch.load('/root/checks_recognize_v2/pths/rotate.pth'))

    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()

    imgs_dir = '/mnt/test_rotate90/images'

    with open('/root/last_dataset/test_result/test_rotate.txt', 'w') as writer:
        for root, dirs, files in os.walk(imgs_dir):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                img_name = file
                img_dir = file_path
                origin_img = Image.open(img_dir).convert('L')

                width = origin_img.width
                height = origin_img.height

                img = origin_img.resize((224, 224))
                img = np.array(img)
                img = img / 255
                img = torch.from_numpy(img).float()
                img = torch.unsqueeze(img, 0)
                img = torch.unsqueeze(img, 0)
                img = img.to(device)
                output = model(img)

                if output[0][0] < output[0][1]:
                    writer.write(img_name + ';')
                    writer.write('1')
                    writer.write(str(width) + ',' + str(height))
                    writer.write('\n')
                else:
                    writer.write(img_name + ';')
                    writer.write('0')
                    writer.write(str(width) + ',' + str(height))
                    writer.write('\n')
                    origin_img.rotate(180, Image.BILINEAR)
                origin_img.save(os.path.join('/root/last_dataset/test_result/test_rotated', img_name))
        writer.close()
