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
    ssim_value, _ = ssim(original, reconstructed, full=True, multichannel=True)
    return ssim_value

# 加载彩色图像 autodl-tmp/NSF/outputs/reflection-dehaze-book-final/combined.png
original_image = cv2.imread('outputs/reflection-dehaze-book-final/reference.png')
reconstructed_image = cv2.imread('outputs/reflection-dehaze-book-final/combined.png')

# 确保图像的大小相同
if original_image.shape != reconstructed_image.shape:
    print("Error: Input images must have the same dimensions!")
else:
    # 计算每个通道的PSNR和SSIM
    psnr_values = []
    ssim_values = []

    # 对每个颜色通道计算PSNR和SSIM
    for i in range(3):  # 0: Blue, 1: Green, 2: Red
        psnr_value = calculate_psnr(original_image[:, :, i], reconstructed_image[:, :, i])
        ssim_value = calculate_ssim(original_image[:, :, i], reconstructed_image[:, :, i])
        
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    # 计算平均PSNR和SSIM
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
