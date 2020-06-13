import config 
import torch 
import random 
import os 
import numpy as np 
import torch.backends.cudnn as cudnn 
import dataset 
from torch.autograd import Variable 
from models import model
import torch.optim as optim 
from tensorboardX import SummaryWriter
import pandas as pd 
from PIL import Image 
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader
import alphabets
import convert
import math
import cv2
from torch.nn import CTCLoss
import torch.optim as optim
import torch.nn.functional as F 


def recog(img, model, converter, device, mode):

    model = model.to(device)
    img = dataset.img_nomalize(img, mode)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    preds = model(img)
    preds = preds.to('cpu')
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = torch.IntTensor([preds.size(0)])
    text = converter.decode(preds.data, preds_size.data, raw=False)
    return text


def select_mode(mode):
    if mode == 'handword':
        return '手写汉字'

    elif mode == 'handnum':
        return '手写数字'

    elif mode == 'word':
        return '印刷汉字'

    elif mode == 'num':
        return '印刷数字'

    elif mode == 'char':
        return '字符'

    elif mode == 'seal':
        return '印章汉字'

    else:
        return '联合汉字'


def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def rotate_image(w, h, angle, scale=1):
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))

    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    # 仿射变换
    return rot_mat


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


def rotate(line, img_name):
    img_name = img_name[:-7] + '.jpg'

    flag = int(rotate_dict[img_name][0])
    vertice = [int(vt) for vt in line[1:-1]]
    vertice = np.array(vertice)
    width = int(rotate_dict[img_name][1:].split(',')[0])
    height = int(rotate_dict[img_name][1:].split(',')[1])
    if flag == 0:
        #print('origin: ', vertice)
        center_x = (width - 1) / 2
        center_y = (height - 1) / 2
        new_vertice = np.zeros(vertice.shape)
        new_vertice[:] = rotate_vertices(vertice, - math.pi, np.array([[center_x], [center_y]]))
        vertice = new_vertice
        #print('rotate: ', vertice)

    flag90 = rotate90_dict[img_name]
    if flag90 == 1:
        rot_mat = rotate_image(width, height, -90, 1)
        newer_vertice = rotate_ver(np.array(vertice), rot_mat)
        vertice = newer_vertice

    vertice = [str(int(vt)) for vt in vertice]
    return vertice


rotate_dict = {}
rotate90_dict = {}

if __name__ =='__main__':

    with open('/root/last_dataset/test_result/test_rotate180.txt', 'r') as reader:
        rotatelines = reader.readlines()
        for rotateline in rotatelines:
            rotateline = rotateline.split(';')
            rotate_dict[rotateline[0]] = rotateline[1][:-1]

    with open('/mnt/test_rotate90/test_rotate90.txt', 'r') as reader90:
        rotatelines90 = reader90.readlines()
        for rotateline90 in rotatelines90:
            rotateline90 = rotateline90.split(';')
            rotate90_dict[rotateline90[0]] = int(rotateline90[1][:-1])

    predict_path = '/root/test_kuangda/predict_last_6.txt'

    if os.path.exists(predict_path):
        os.remove(predict_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


    alphabetdict = {'handword': alphabets.alphabet_handword, 'handnum': alphabets.alphabet_handnum,
                    'word': alphabets.alphabet_word, 'num': alphabets.alphabet_num, 'char': alphabets.alphabet_char,
                    'seal': alphabets.alphabet_word, 'catword': alphabets.alphabet_word}

    modeldict = {'handword': 'handword_lr3_epoch14_acc0.936107.pth', 'handnum': 'handnum_lr3_epoch32_acc0.972464.pth',
                 'word': 'word_lr3_bat128_expaug_epoch34_acc0.951470.pth', 'num': 'num_lr3_epoch22_acc0.900240.pth', 'char': 'char_lr3_epoch10_acc0.997729.pth',
                 'seal': 'seal_lr3_bat256_expaug_epoch64_acc0.860740.pth', 'catword': 'catword_lr3_epoch_16_acc0.585213.pth'}


    for i, mode in enumerate(['handword', 'handnum', 'num', 'word', 'char', 'catword', 'seal']):

        print('mode :', mode)
        alphabet = alphabetdict[mode]
        n_class = len(alphabet) + 1

        converter = convert.strLabelConverter(alphabet)
        now_model = model.CRNN(class_num=n_class, backbone='resnet', pretrain=False)

        state_dict = torch.load(os.path.join('/root/last_dataset/last_pths', modeldict[mode]))
        now_model.load_state_dict(state_dict=state_dict)

        now_model.to(device)
        now_model.eval()

        chn_mode = select_mode(mode)
        print('chn_mode ', chn_mode)

        testlines = []
        test_mode = []

        if i < 6:
            test_mode = ['test_crop2']
        elif i == 6:
            test_mode = ['test_stamp_crop']

        for txtmode in test_mode:
            txt_dir = os.path.join('/root/last_dataset/test_result', txtmode+'.txt')
            with open(txt_dir, 'r') as lb:
                lines = lb.readlines()
                for line in lines:
                    line = line.split(';')
                    if line[-1][:-1] == chn_mode:
                        anno = {}
                        anno['img_name'] = line[0]
                        rotate_box = rotate(line, line[0])
                        anno['box'] = rotate_box
                        anno['mode'] = txtmode
                        testlines.append(anno)
        '''
        for x in testlines[:30]:
            print('img_name', x['img_name'])
            print('box', x['box'])
            print('mode', x['mode'])
        '''

        with open(predict_path, 'a') as writer:
            for l, imgline in enumerate(testlines):
                img_name = imgline['img_name']
                img_mode = imgline['mode']
                img_dir = os.path.join('/root/last_dataset/test_result', img_mode, img_name)

                img = Image.open(img_dir).convert("L")
                result = recog(img, now_model, converter, device, mode)
                if mode == 'seal' and (result.find('财') != -1 or result.find('章') != -1):
                    result = '财务专用章'

                if mode == 'char' and result != '√':
                    result = '×'

                if mode == 'catword':
                    result = result[:-10] + '   ' + result[-10:]
                '''
                if mode == 'seal':
                    result.replace('(', '')

                if mode == 'catword':
                    result.replace('(', '')

                if mode == 'word':
                    result.replace('(', '')

                    if len(result) == 2:
                        if result.find('工') != -1 or result.find('资') != -1:
                            result = '工资'
                        if result.find('贷') != -1 or result.find('款') != -1:
                            result = '贷款'
                        if result.find('电') != -1 or result.find('费') != -1:
                            result = '电费'
                    if len(result) == 3:
                        if result.find('代') != -1 or result.find('理') != -1:
                            result = '代理费'
                        if result.find('差') != -1 or result.find('款') != -1:
                            result = '差旅款'
                        if result.find('备') != -1 or result.find('金') != -1:
                            result = '备用金'
                        if result.find('物') != -1 or result.find('业') != -1:
                            result = '物业费'
                        if result.find('往') != -1 or result.find('来') != -1:
                            result = '往来款'
                        if result.find('装') != -1 or result.find('修') != -1:
                            result = '装修费'
                        if result.find('劳') != -1 or result.find('务') != -1:
                            result = '劳务费'
                    if len(result) == 5:
                        if result.find('备') != -1 or result.find('公') != -1:
                            result = '公司筹备金'
                '''
                #save_path = os.path.join("test", img_name.split(".")[0] + "_" + result + ".jpg")
                #img.save(save_path)

                writer.write(img_name[:-7] + '.jpg' + ';')
                box = ';'.join(imgline['box'])
                writer.write(box + ';')
                writer.write(result + '\n')
    
            writer.close()










