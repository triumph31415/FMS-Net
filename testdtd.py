import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math

# 计算PSNR值
def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100  # 如果没有误差（完全相同），PSNR为100
    max_pixel = 255.0  # 假设图像是8位（像素值范围0-255）
    psnr_value = 10 * math.log10((max_pixel ** 2) / mse)
    return psnr_value

# 计算SSIM值
def calculate_ssim(original, reconstructed):
    ssim_value, _ = ssim(original, reconstructed, full=True)
    return ssim_value

# 加载图像（灰度图像）
original_image = cv2.imread('outputs/reflection-dehaze-book-final/reference.png', cv2.IMREAD_GRAYSCALE)
reconstructed_image = cv2.imread('outputs/reflection-dehaze-book-final/combined.png', cv2.IMREAD_GRAYSCALE)

# 确保图像的大小相同
if original_image.shape != reconstructed_image.shape:
    print("Error: Input images must have the same dimensions!")
else:
    # 计算PSNR和SSIM
    psnr_value = calculate_psnr(original_image, reconstructed_image)
    ssim_value = calculate_ssim(original_image, reconstructed_image)

    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
