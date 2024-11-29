import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def extract_first_elements_from_txt_files(folder_path):
    width = 9
    height = 9

    # 存储每个文件中每行的第一个元素（从第二行开始）
    all_elements = {}

    # 遍历指定文件夹中的所有txt文件
    for filename_ in os.listdir(folder_path):
        data = {}
        pt1 = []
        pt2 = []

        if filename_.endswith("result.txt"):
            file_path = os.path.join(folder_path, filename_)
            with open(file_path, 'r') as file:
                lines = file.readlines()  # 读取所有行
                keypoint_line = [lines[1].split(','), lines[height].split(','), lines[(width - 1) * height].split(','),
                                 lines[width * height].split(','), lines[int((width * height) / 4)].split(',')]

                for line in keypoint_line:
                    pt1.append([float(line[0]) + float(line[2]), float(line[1]) + float(line[3])])
                    pt2.append([float(line[0]) + float(line[4]), float(line[1]) + float(line[5])])

                data['pt1'] = pt1
                data['pt2'] = pt2

        all_elements[filename_] = data
    return all_elements


# 按图片顺序重建
def reconstruction(pts1, pts2):
    K1 = np.array([[1.3811813910601268e+04, 0, 1.5926784651273513e+03],
                   [0, 1.3821384261602509e+04, 1.3311098752645803e+03],
                   [0, 0, 1]])

    K2 = np.array([[1.3702106800557507e+04, 0, 1.5362739415608321e+03],
                   [0, 1.3714187583617171e+04, 1.2510139215570434e+03],
                   [0, 0, 1]])

    # 假设你的外参是 R1, t1 和 R2, t2
    # R 是 3x3 的旋转矩阵，t 是 3x1 的平移向量
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    R2 = np.array([[9.8648630752151134e-01, -5.3398561583436616e-03, 1.6375668233321777e-01],
                   [4.7652332241665922e-03, 9.9998103474813227e-01, 3.9016274007045611e-03],
                   [-1.6377441077559551e-01, -3.0685632245122533e-03, 9.8649304422020323e-01]])

    t2 = np.array([[-8.9662054045351454e+02, -1.4723530268386430e+01, 3.6297616612887019e+01]])
    # R2 = np.array([[0.986731, -0.00525905, 0.162281],
    #                [0.00471448, 0.999982, 0.00374064],
    #                [-0.162298, -0.00292593, 0.986737]])
    # t2 = np.array([[-906.86, -16.5388, 53.485]])
    R2 = np.array([[0.986984, -0.00458998, 0.160756],
                   [0.00393082, 0.999983, 0.00441814],
                   [-0.160773, -0.00372873, 0.986984]])
    t2 = np.array([[-897.786, -29.8904, 68.9631]])
    # 计算投影矩阵 P1 和 P2
    P1 = K1 @ np.hstack((R1, t1))  # 3x4 投影矩阵
    P2 = K2 @ np.hstack((R2, t2.T))  # 3x4 投影矩阵

    pts1 = pts1.T
    pts2 = pts2.T

    # 使用 OpenCV 的三角测量函数计算3D点
    points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # 将齐次坐标转换为非齐次坐标
    points_3D = points_4D[:3] / points_4D[3]  # 3xN 的非齐次3D点坐标

    # 转置为 Nx3 格式的3D点
    points_3D = points_3D.T
    return points_3D


def calculate_dis(points_3D):
    base_pt = points_3D[1, :]
    displacement = []
    for pts in points_3D:
        temp_dis = []
        dis = pts - base_pt
        for d in dis:
            temp_dis.append(np.sqrt(np.sum(d ** 2)))
        displacement.append(temp_dis)
    return displacement


if __name__ == '__main__':
    # 使用示例，指定包含txt文件的文件夹路径
    folder_path = "D:/SoftwareQt/DualDIC/1104/DICResult"
    first_elements = extract_first_elements_from_txt_files(folder_path)

    displacement = []
    pts_3D = []
    # 打印提取的元素
    for filename, element in first_elements.items():
        if filename.endswith("result.txt"):
            pts_3D.append(reconstruction(pts1=np.array(element['pt1']), pts2=np.array(element['pt2'])))
    displacement = calculate_dis(np.array(pts_3D))
    displacement = np.array(displacement)
    # print(displacement)
    for ele in displacement:
        print(ele.mean() - 50)

    # Generate the plot
    plt.figure(figsize=(10, 6))
    for i in range(displacement.shape[1]):
        plt.plot(displacement[:, i], label=f'Series {i + 1}')

    # Add labels and legend
    plt.title("Line Plot of Provided Data Series")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
