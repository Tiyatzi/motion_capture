import os
import cv2
import copy
import numpy as np


# 椭圆拟合及筛选
def ellipse_fit_filter(image, contours):
    """
    Args:
        image (_type_): init image
        contours (_type_): canny edge detection
    """

    # save circle and bounding box
    result = []

    # detect circle
    processed_centers = []

    for contour in contours:
        # 如果轮廓点数不足 5，则跳过
        if len(contour) < 5:
            continue

        # 根据圆的面积筛选
        area = cv2.contourArea(contour)
        if area < 10 or area > 50:
            continue

        # 根据圆形度筛选   
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # 圆形度应接近 1，设置合适的阈值
        if circularity < 0.9:
            continue
        # 椭圆拟合
        ellipse = cv2.fitEllipse(contour)

        # 计算轮廓的几何矩
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            continue

        # 合并相近的圆心
        close_enough = False
        for center in processed_centers:
            if np.linalg.norm(np.array(center) - np.array((cx, cy))) < 10:  # 设定距离阈值
                close_enough = True
                break

        if close_enough:
            continue  # 如果相近则跳过

        # 拟合椭圆
        ellipse = cv2.fitEllipse(contour)
        w, h = ellipse[1][0], ellipse[1][1]

        # 计算 bounding box 的左上角坐标
        top_left_x = int(cx - w / 2)
        top_left_y = int(cy - h / 2)

        # 在图像上绘制 bounding box
        cv2.rectangle(image, (top_left_x, top_left_y), (top_left_x + int(w), top_left_y + int(h)), (255, 0, 0), 2)

        # 标注圆心
        int_center = (int(cx), int(cy))
        center = (cx, cy)
        cv2.circle(image, int_center, 3, (0, 0, 255), -1)

        # 记录处理后的圆心
        processed_centers.append((cx, cy))
        result.append((cx, cy, w, h))
        # 输出bounding box的宽高和圆心坐标
        # print(f"Center: {center}, Bounding Box (w, h): ({w}, {h})")

    # # 显示结果
    # cv2.imshow('Fitted Ellipses', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return result


# Calculate normalized coordinates and store in label.txt
def coordinate_norm_and_save(bounding_boxes, image_width, image_height, save_path, image_filename):
    # 存储归一化的 bounding box
    normalized_bounding_boxes = []

    for bbox in bounding_boxes:
        # 计算归一化的中心坐标
        x_center_normalized = bbox[0] / image_width
        y_center_normalized = bbox[1] / image_height

        # 计算归一化的宽度和高度
        w_normalized = bbox[2] / image_width
        h_normalized = bbox[3] / image_height

        # 将归一化结果存储在列表中
        normalized_bounding_boxes.append({
            'class_id': 0,
            'x_center': x_center_normalized,
            'y_center': y_center_normalized,
            'width': w_normalized,
            'height': h_normalized
        })

    # 创建标签文件路径
    label_filename = os.path.join(save_path, f"{image_filename}.txt")

    # 写入归一化结果到文本文件
    with open(label_filename, 'w') as f:
        for bbox in normalized_bounding_boxes:
            # 按要求格式写入每一行
            line = f"{bbox['class_id']} {bbox['x_center']} {bbox['y_center']} {bbox['width']} {bbox['height']}\n"
            f.write(line)


def circle_detect(image):
    iteration = 3

    threshold_difference = 60

    for i in range(iteration):
        # 转为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # 使用 Canny 进行边缘检测
        edges = cv2.Canny(blurred, threshold_difference, 180)

        # 显示边缘检测后的图像
        # cv2.imshow("Edges Detected", edges)

        # 等待用户按键
        # cv2.waitKey(0)  # 等待按键才能继续，0表示无限等待

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return ellipse_fit_filter(image, contours)


video_paths = [
    # 'video_data\TC_S1_acting1_cam1.mp4',
    # 'video_data\TC_S1_acting1_cam2.mp4',
    # 'video_data\TC_S1_acting1_cam3.mp4',
    # 'video_data\TC_S1_acting1_cam4.mp4',
    # 'video_data\TC_S1_acting1_cam5.mp4',
    # 'video_data\TC_S1_acting1_cam6.mp4',
    'video_data\\TC_S1_acting1_cam7.mp4',
    'video_data\\TC_S1_acting1_cam8.mp4'
]

img_save_paths = 'E:\\INFJ\\maker_data\\images'
paint_img_save_path = 'data\\circle_images'
labels_save_path = 'E:\\INFJ\\maker_data\\labels'

close_video = False

for video_path in video_paths:

    video_name = video_path.split('\\')[1].split('.')[0]

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    cv2.namedWindow("Frame")

    index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        paint_img = copy.deepcopy(frame)

        center_bounding_box = circle_detect(paint_img)

        cv2.imshow("Frame", paint_img)

        if len(center_bounding_box) > 5:
            cv2.imwrite(img_save_paths + '\\' + video_name + '_' + str(index) + '.bmp', frame)
            # cv2.imwrite(paint_img_save_path + '\\' + video_name + '_' + str(index) + '.bmp', paint_img)
            coordinate_norm_and_save(center_bounding_box, frame.shape[1], frame.shape[0], labels_save_path,
                                     video_name + '_' + str(index))
            print('图片和标签保存成功')
            index += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_video = True
            break
        # # 检测键盘输入
        # key = cv2.waitKey(0) & 0xFF
        # if key == ord('c'):  # 按下 'c' 表示接受当前帧
        #     print("圆心检测通过，处理当前帧...")
        #     # 可以在此保存处理结果或进行其他操作
        #     cv2.imwrite(imgs_save_path + '\\' + video_name + '_' + str(index) + '.bmp', frame)
        #     cv2.imwrite(paint_img_save_path + '\\' + video_name + '_' + str(index) + '.bmp', paint_img)
        #     coordinate_norm_and_save(center_bounding_box, frame.shape[1], frame.shape[0], labels_save_path, video_name + '_' + str(index))
        #     print('图片和标签保存成功')
        #     index += 1

        # elif key == ord('d'):  # 按下 'd' 表示跳过当前帧
        #     print("跳过当前帧")

        # elif key == ord('q'):  # 按下 'q' 退出
        #     break

    if close_video:
        break

    cap.release()
    cv2.destroyAllWindows()
