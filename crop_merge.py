import os
from PIL import Image

if __name__ == '__main__':
    mode1_dir = '/mnt/train_rotate180/'
    mode2_dir = '/mnt/valid_rotate180/'
    #mode3_dir = '/home/chen-ubuntu/Desktop/checks_dataset/valid_crop_mode3'

    dirlist = [mode1_dir, mode2_dir]#, mode3_dir]

    merge_dir = '/mnt/tvmerge_rotate180/'

    for i in range(2):
        img_dir = os.path.join(dirlist[i], 'images')
        img_files = os.listdir(img_dir)
        for img_file in sorted(img_files):
            img_path = os.path.join(img_dir, img_file)
            img = Image.open(img_path)
            new_path = os.path.join(merge_dir, 'images', img_file)
            print(new_path)
            img.save(new_path)

    for i in range(2):
        txt_dir = os.path.join(dirlist[i], 'labels')
        txt_files = os.listdir(txt_dir)
        for txt_file in sorted(txt_files):
            txt_path = os.path.join(txt_dir, txt_file)
            #img = Image.open(img_path)
            new_path = os.path.join(merge_dir, 'labels', txt_file)
            print(new_path)
            #img.save(new_path)
            with open(new_path, 'w') as writer:
                with open(txt_path, 'r') as lines:
                    lines = lines.readlines()
                    for l, line in enumerate(lines):
                        writer.write(line)
            writer.close()
