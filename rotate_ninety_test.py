import os
import numpy as np
from PIL import Image
import math
import cv2

createImg_root = '/mnt/test_rotate90/images'
#createLab_root = '/mnt/train_rotate90/labels'

def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def rotate_image(src, angle, scale=1):
    w = src.shape[1]
    h = src.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))

    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    dst = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4, borderMode=0,
                         borderValue=(255, 255, 255))
    # 仿射变换
    return dst, rot_mat


def rotate_ver(box, rot_mat):
    box = box.reshape(4, 2)
    point1 = np.dot(rot_mat, np.array([box[0, 0], box[0, 1], 1]))  # 获取原始矩形的四个中点，然后将这四个点转换到旋转后的坐标系下
    point2 = np.dot(rot_mat, np.array([box[1, 0], box[1, 1], 1]))
    point3 = np.dot(rot_mat, np.array([box[2, 0], box[2, 1], 1]))
    point4 = np.dot(rot_mat, np.array([box[3, 0], box[3, 1], 1]))
    concat = np.vstack((point1, point2, point3, point4))  # 合并np.array
    concat = concat.astype(np.int32)
    new_box = concat
    new_box = new_box.reshape(-1)
    return np.array(new_box)


'''
def rotate_img(img_dir, img_savepath, file_path, lab_savepath, flag):
    image = Image.open(img_dir)
    if flag:
        image = image.rotate(90, expand=1)
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
                    new_center_x = (image.height - 1) / 2
                    new_center_y = (image.width - 1) / 2
                    inc_x = []
                    for x in vertice[0::2]:
                        tmpx = x - center_x
                        inc_x.append(tmpx)
                    inc_y = []
                    for y in vertice[1::2]:
                        tmpy = y - center_y
                        inc_y.append(tmpy)
                    new_vertice = []
                    for t, vt in enumerate(vertice):
                        if t % 2 == 0:
                            new_vertice.append(new_center_x + inc_y[t//2])
                            print('vt, + ', t)
                        else:
                            new_vertice.append(new_center_y + inc_x[t//2])
                            print('vt, - ', t)

                    inc = (image.height - image.width) / 2
                    new_vertice = np.zeros(vertice.shape)
                    new_vertice[:] = rotate_vertices(vertice, - math.pi/2, np.array([[center_x], [center_y]]))

                    inc_vertice = []
                    for t, vt in enumerate(new_vertice):
                        if t % 2 == 0:
                            inc_vertice.append(vt + inc)
                            print('vt, + ', t)
                        else:
                            inc_vertice.append(vt - inc)
                            print('vt, - ', t)
                    vertice = inc_vertice

                    vertice = new_vertice
                new_line = []
                new_line.append(line[0])
                for v in vertice:
                    new_line.append(str(int(v)))
                new_line.append(line[-1])
                new_line = ';'.join(new_line)
                writer.write(new_line)
        writer.close()
'''

if __name__ == '__main__':

    img_root_dir = '/mnt/data/datasource/test/images'
    with open('/mnt/test_rotate90/test_rotate90.txt', 'w') as writer:
        for image_name in os.listdir(img_root_dir):
            image_dir = os.path.join(img_root_dir, image_name)
            image = Image.open(image_dir)
            w, h = image.size
            #flag = True
            new_line = []
            new_line.append(image_name)
            if w > h:
                #flag = False
                new_line.append('0\n')# 没有转90度
                print('no 90')
            else:
                new_line.append('1\n')  # 转了90度
            new_line = ';'.join(new_line)
            writer.write(new_line)
        writer.close()
            #img_savepath = os.path.join(createImg_root, image_name)
            #if flag:
            #    image, rot_mat = rotate_image(image, 90, scale=1)
            #cv2.imwrite(img_savepath, image)

