import cv2
import numpy as np


def undistort_image(image_path, camera_matrix, dist_coeffs):
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("无法加载图片，请检查路径。")
        return None

    # 获取图像尺寸
    h, w = image.shape[:2]

    # 计算新的相机矩阵以减少图像裁剪
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # 矫正畸变
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # 裁剪图像到有效区域
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y + h, x:x + w]

    return undistorted_image


# 示例：相机内参和畸变系数（需要根据你的相机校准数据进行修改）
# 左
# camera_matrix = np.array([[1.3811813910601268e+04, 0, 1.5926784651273513e+03], [0, 1.3821384261602509e+04, 1.3311098752645803e+03], [0, 0, 1]], dtype=np.float32)
# dist_coeffs = np.array([-1.2678423857708579e-01, -7.5779864232656990e-01, 0, 0, 0], dtype=np.float32)
# 右
camera_matrix = np.array([[1.3702106800557507e+04, 0, 1.5362739415608321e+03], [0, 1.3714187583617171e+04, 1.2510139215570434e+03], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([-9.3938125188720786e-01, -1.2487292873079856e-01, 0, 0, 0], dtype=np.float32)

# 使用示例
image_path = "D:/SoftwareQt/DualDIC/1104/marker/marker_1.bmp"
corrected_image = undistort_image(image_path, camera_matrix, dist_coeffs)

# 显示结果
if corrected_image is not None:
    # cv2.imshow("Undistorted Image", corrected_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 或者保存矫正后的图片
    cv2.imwrite("undistorted_marker_1.jpg", corrected_image)
