# （基于透视的图像矫正）
import cv2
import opencv_test as mycv2
from matplotlib import pyplot as plt
import os
import numpy as np


if __name__ == "__main__":

    trainImgPath = '/root/befo_dataset/train/train/one/Image'
    trainLabelPath = '/root/befo_dataset/train/train/one/Label'

    for root, dirs, files in os.walk(trainLabelPath):
        for file in files[: 1]:
            file_path = os.path.join(root, file)
            print(file_path)

            image_name = file[0: -4] + '.jpg'
            image_path = os.path.join(trainImgPath, image_name)

            print(image_path)

            original_img, gray_img, RedThresh, closed, opened = mycv2.Img_Outline(image_path)
            unbox, draw_img = mycv2.findContours_img(original_img, opened)
            draw_img = cv2.drawContours(draw_img, [unbox], 0, (0, 0, 255), 3)

            origin_boxes = []
            with open(file_path, 'r') as lines:
                lines = lines.readlines()
                for line in lines:
                    box = []
                    sites = line.split(';')
                    xlist = []
                    for site in sites[1: 9: 2]:
                        xlist.append(site)
                    ylist = []
                    for site in sites[2: 9: 2]:
                        ylist.append(site)
                    for i in range(4):
                        box.append([int(xlist[i]), int(ylist[i])])
                    origin_boxes.append(box)
                    box = np.array(box)
                    draw_img = cv2.polylines(draw_img, [box], True, (0, 255, 0))

            #cv2.imshow("draw_img", draw_img)
            cv2.imwrite(os.path.join('/root/tmp_test/', '1_'+image_name), draw_img)

            result_img, result_boxes = mycv2.Perspective_train_transform(unbox, origin_boxes, original_img)

            for result_box in result_boxes:
                print('result_box: ', result_box)
                result_box = np.array(result_box, dtype='int32')
                result_img = cv2.polylines(result_img, [result_box], True, (0, 255, 0))

            #cv2.imshow("result_img", result_img)
            cv2.imwrite(os.path.join('/root/tmp_test/', '2_'+image_name), result_img)

            predict_boxes = mycv2.Perspective_predict_transform(unbox, result_boxes)

            for predict_box in predict_boxes:
                print('result_box: ', predict_box)
                predict_box = np.array(predict_box, dtype='int32')
                predict_img = cv2.polylines(original_img, [predict_box], True, (0, 255, 0))

            #cv2.imshow("predict_img", predict_img)
            cv2.imwrite(os.path.join('/root/tmp_test/', '3_'+image_name), predict_img)

            #cv2.waitKey(0)
            #cv2.destroyAllWindows()



