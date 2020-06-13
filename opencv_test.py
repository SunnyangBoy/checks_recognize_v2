# （基于透视的图像矫正）
import cv2
import math
import numpy as np

def Img_Outline(input_dir):
    original_img = cv2.imread(input_dir)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 增强对比度
    emhance_img = cv2.equalizeHist(gray_img)
    #cv2.imshow("emhance", emhance_img)

    blurred = cv2.GaussianBlur(emhance_img, (9, 9), 0)  # 高斯模糊去噪（设定卷积核大小影响效果）
    _, RedThresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY_INV)  # 设定阈值165（阈值影响开闭运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)  # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)  # 开运算（去噪点）
    return original_img, gray_img, emhance_img, RedThresh, closed, opened

def findContours_img(original_img, opened):
    original_img = original_img.copy()
    vertices = []

    height, width = original_img.shape[:2]
    upleft_lr = int(-1/(height/width))
    upright_lr = int(-1/(-height/width))
    dowleft_lr = upright_lr
    dowright_lr = upleft_lr

    vertice1 = (0, 0)
    for h in range(height+1000):
        flag = 1
        for x in range(width):
            y = (upleft_lr * x) + h
            if y < height and y > 0:
                if opened[y][x] > 0:
                    vertice1 = (x, y)
                    flag = 0
                    break
        if flag == 0:
            break
    print('vertice1: ', vertice1)
    vertices.append(vertice1)
    draw_img = cv2.circle(original_img, vertice1, 5, [0, 0, 255])


    vertice2 = (width-1, 0)
    for h in range(height+1000):
        flag = 1
        for x in range(width):
            y = h - (upright_lr * x)
            if y < height and y > 0:
                if opened[y][width-1-x] > 0:
                    vertice2 = (width-1-x, y)
                    flag = 0
                    break
        if flag == 0:
            break
    print('vertice2: ', vertice2)
    vertices.append(vertice2)
    draw_img = cv2.circle(draw_img, vertice2, 5, [0, 0, 255])


    vertice4 = (width-1, height-1)
    for h in range(-1000, height)[::-1]:
        flag = 1
        for x in range(width):
            y = h - (dowright_lr * x)
            if y < height and y > 0:
                if opened[y][width-1-x] > 0:
                    vertice4 = (width - 1 - x, y)
                    flag = 0
                    break
        if flag == 0:
            break
    print('vertice4: ', vertice4)
    vertices.append(vertice4)
    draw_img = cv2.circle(draw_img, vertice4, 5, [0, 0, 255])


    vertice3 = (0, height-1)
    for h in range(-1000, height)[::-1]:
        flag = 1
        for x in range(width):
            y = (dowleft_lr * x) + h
            if y < height and y > 0:
                if opened[y][x] > 0:
                    vertice3 = (x, y)
                    flag = 0
                    break
        if flag == 0:
            break
    print('vertice3: ', vertice3)
    vertices.append(vertice3)
    draw_img = cv2.circle(draw_img, vertice3, 5, [0, 0, 255])

    vertices.append(vertice1)
    return np.array(vertices), draw_img


def Perspective_transform(box, original_img):
    # 获取画框宽高(x=orignal_W, y=orignal_H)
    orignal_W1 = math.ceil(np.sqrt((box[3][1] - box[2][1]) ** 2 + (box[3][0] - box[2][0]) ** 2))
    orignal_W2 = math.ceil(np.sqrt((box[1][1] - box[0][1]) ** 2 + (box[1][0] - box[0][0]) ** 2))
    orignal_W = orignal_W1 if orignal_W1 > orignal_W2 else orignal_W2

    orignal_H1 = math.ceil(np.sqrt((box[3][1] - box[0][1]) ** 2 + (box[3][0] - box[0][0]) ** 2))
    orignal_H2 = math.ceil(np.sqrt((box[2][1] - box[1][1]) ** 2 + (box[2][0] - box[1][0]) ** 2))
    orignal_H = orignal_H1 if orignal_H1 > orignal_H2 else orignal_H2

    # 原图中的四个顶点,与变换矩阵
    pts1 = np.float32([box[2], box[3], box[0], box[1]])
    pts2 = np.float32(
        [[int(orignal_W + 1), int(orignal_H + 1)], [0, int(orignal_H + 1)], [0, 0], [int(orignal_W + 1), 0]])

    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)

    result_img = cv2.warpPerspective(original_img, M, (int(orignal_W + 3), int(orignal_H + 1)))

    return result_img


def Perspective_train_transform(box, origin_boxes, original_img):
    # 获取画框宽高(x=orignal_W, y=orignal_H)
    orignal_W1 = math.ceil(np.sqrt((box[3][1] - box[2][1]) ** 2 + (box[3][0] - box[2][0]) ** 2))
    orignal_W2 = math.ceil(np.sqrt((box[1][1] - box[0][1]) ** 2 + (box[1][0] - box[0][0]) ** 2))
    orignal_W = orignal_W1 if orignal_W1 > orignal_W2 else orignal_W2

    orignal_H1 = math.ceil(np.sqrt((box[3][1] - box[0][1]) ** 2 + (box[3][0] - box[0][0]) ** 2))
    orignal_H2 = math.ceil(np.sqrt((box[2][1] - box[1][1]) ** 2 + (box[2][0] - box[1][0]) ** 2))
    orignal_H = orignal_H1 if orignal_H1 > orignal_H2 else orignal_H2

    # 原图中的四个顶点,与变换矩阵
    pts1 = np.float32([box[2], box[3], box[0], box[1]])
    pts2 = np.float32([[int(orignal_W + 1), int(orignal_H + 1)], [0, int(orignal_H + 1)], [0, 0], [int(orignal_W + 1), 0]])

    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    #print('M: ', M)

    result_img = cv2.warpPerspective(original_img, M, (int(orignal_W + 1), int(orignal_H + 1)))

    #print(origin_boxes)
    result_boxes = []
    for origin_box in origin_boxes:
        tmp_box = np.array(origin_box, dtype='float32')
        tmp_box = np.array([tmp_box])
        result_box = cv2.perspectiveTransform(tmp_box, M)
        result_box = np.squeeze(result_box, 0)
        result_boxes.append(result_box)
    return result_img, result_boxes


def Perspective_predict_transform(box, det_boxes):
    # 获取画框宽高(x=orignal_W, y=orignal_H)
    orignal_W1 = math.ceil(np.sqrt((box[3][1] - box[2][1]) ** 2 + (box[3][0] - box[2][0]) ** 2))
    orignal_W2 = math.ceil(np.sqrt((box[1][1] - box[0][1]) ** 2 + (box[1][0] - box[0][0]) ** 2))
    orignal_W = orignal_W1 if orignal_W1 > orignal_W2 else orignal_W2

    orignal_H1 = math.ceil(np.sqrt((box[3][1] - box[0][1]) ** 2 + (box[3][0] - box[0][0]) ** 2))
    orignal_H2 = math.ceil(np.sqrt((box[2][1] - box[1][1]) ** 2 + (box[2][0] - box[1][0]) ** 2))
    orignal_H = orignal_H1 if orignal_H1 > orignal_H2 else orignal_H2

    # 原图中的四个顶点,与变换矩阵
    pts1 = np.float32([box[2], box[3], box[0], box[1]])
    pts2 = np.float32([[int(orignal_W + 1), int(orignal_H + 1)], [0, int(orignal_H + 1)], [0, 0], [int(orignal_W + 1), 0]])

    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts2, pts1)
    #print('M: ', M)

    result_boxes = []
    for det_box in det_boxes:
        tmp_box = np.array(det_box, dtype='float32')
        tmp_box = np.array([tmp_box])
        result_box = cv2.perspectiveTransform(tmp_box, M)
        result_box = np.squeeze(result_box, 0)
        result_boxes.append(result_box)
    return result_boxes



if __name__ == "__main__":
    input_dir = "/Users/zhuzhenyang/Downloads/train/one/Image/receipt_img_tm1_08182.jpg"
    original_img, gray_img, emhance_img, RedThresh, closed, opened = Img_Outline(input_dir)
    box, draw_img = findContours_img(original_img, opened)
    draw_img = cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 3)

    
    #minLineLength = 50
    #maxLineGap = 100
    ##lines = cv2.HoughLinesP(opened, 0.8, np.pi / 180, 10, minLineLength, maxLineGap)

    #lines = cv2.HoughLinesP(opened, 1, np.pi / 180, 80, minLineLength, maxLineGap)
    #for line in lines:
    #    x1, y1, x2, y2 = line[0]
    #    cv2.line(opened, (x1, y1), (x2, y2), (0, 255, 0), 2)


    #lines = cv2.HoughLines(opened, 1, np.pi / 180, 100)  # 这里对最后一个参数使用了经验型的值
    #for line in lines:
    #    rho = line[0][0]  # 第一个元素是距离rho
    #    theta = line[0][1]  # 第二个元素是角度theta
    #    print(rho)
    #    print(theta)
    #    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
    #        pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
    #        # 该直线与最后一行的焦点
    #        pt2 = (int((rho - opened.shape[0] * np.sin(theta)) / np.cos(theta)), opened.shape[0])
    #        cv2.line(opened, pt1, pt2, (255),2)  # 绘制一条白线
    #    else:  # 水平直线
    #        pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
    #        # 该直线与最后一列的交点
    #        pt2 = (opened.shape[1], int((rho - opened.shape[1] * np.cos(theta)) / np.sin(theta)))
    #        cv2.line(opened, pt1, pt2, (255), 2)  # 绘制一条直线
    

    result_img = Perspective_transform(box, original_img)
    cv2.imshow("original", original_img)
    cv2.imshow("RedThresh", RedThresh)
    cv2.imshow("gray", gray_img)
    cv2.imshow("closed", closed)
    cv2.imshow("opened", opened)
    cv2.imshow("draw_img", draw_img)
    cv2.imshow("result_img", result_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

