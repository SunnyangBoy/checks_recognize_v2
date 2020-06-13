import torch
from dataset import DetectAngleDataset
from torch.utils.data import DataLoader
from torch import optim
import configs
from model import DetectAngleModel
import os
import torch.nn as nn


def valid(valid_dataloader, model, device):
    model.eval()
    valid_loss = 0
    acc = 0
    cnt = 0
    for i, (img, img_class) in enumerate(valid_dataloader):
        img = img.to(device)
        image_class = img_class
        img_class = img_class.to(device)
        output = model(img)
        loss = criterion(output, img_class)
        valid_loss += loss
        if output[0][0] < output[0][1]:
            if image_class.item() == 1:
                acc += 1
        if output[0][0] >= output[0][1]:
            if image_class.item() == 0:
                acc += 1
        cnt += 1

    return acc/cnt, valid_loss

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = 500

    avg_imgs = []
    images = []
    cnt = 0
    for root, dirs, files in os.walk(configs.label_dir):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            image_name = file[0: -4] + '.jpg'
            with open(file_path, 'r') as lines:
                lines = lines.readlines()
                first_y = int(lines[0].split(';')[2])
                last_y = int(lines[0].split(';')[-2])
                annotation = {}
                annotation['name'] = image_name
                if first_y < last_y:
                    annotation['class'] = 1
                else:
                    annotation['class'] = 0
                    cnt += 1
                    avg_imgs.append(annotation)
                images.append(annotation)

    print('nag_imgs: ', cnt)
    for gt_img in images:
        if gt_img['class'] == 1:
            avg_imgs.append(gt_img)
            cnt -= 1
            if cnt == 0:
                break
    print('avg_imgs: ', len(avg_imgs))

    train_dataset = DetectAngleDataset(configs.img_rootdir, avg_imgs)
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    valid_imgs = []
    for root, dirs, files in os.walk('/mnt/valid_rotate90/labels'):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            image_name = file[0: -4] + '.jpg'
            with open(file_path, 'r') as lines:
                lines = lines.readlines()
                first_y = int(lines[0].split(';')[2])
                last_y = int(lines[0].split(';')[-2])
                annotation = {}
                annotation['name'] = image_name
                if first_y < last_y:
                    annotation['class'] = 1
                else:
                    annotation['class'] = 0
                valid_imgs.append(annotation)

    valid_dataset = DetectAngleDataset('/mnt/valid_rotate90/images', valid_imgs)
    valid_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    model = DetectAngleModel()

    #model.load_state_dict(torch.load('/home/chen-ubuntu/Desktop/checks_dataset/pths/model_epoch_mode3_14.pth'))

    data_parallel = False
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        data_parallel = True
        model = nn.DataParallel(model)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for i, (img, img_class) in enumerate(dataloader):
            img = img.to(device)
            img_class = img_class.to(device)
            output = model(img)
            loss = criterion(output, img_class)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('epoch {} / {} ： loss: {:4f}'.format(epoch, epochs, epoch_loss))

        if epoch % 5 == 0:
            #acc, valid_loss = valid(valid_dataloader, model, device)
            #print('============== valid ===============')
            #print('valid_acc: ', acc, '   valid_loss: ', valid_loss)
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join('/root/last_dataset/tmp_pths/', 'rotated_{}_{:.4f}.pth'.format(epoch, epoch_loss)))
