import cv2
import numpy as np
import os

def visualize_difference(file_path1, file_path2, output_diff_path='diff_debug.png'):
    # 讀取圖片
    img1 = cv2.imread(file_path1)
    img2 = cv2.imread(file_path2)
    
    if img1 is None or img2 is None:
        print("無法讀取圖片，請確認路徑。")
        return

    if img1.shape != img2.shape:
        print(f"形狀不同，無法進行像素級比對: {img1.shape} vs {img2.shape}")
        return

    # 計算絕對差異
    diff = cv2.absdiff(img1, img2)
    
    # 計算各通道的平均值，看看是否有特定顏色偏差
    mean1 = np.mean(img1, axis=(0,1))
    mean2 = np.mean(img2, axis=(0,1))
    print(f"圖片 1 (BGR 平均值): {mean1}")
    print(f"圖片 2 (BGR 平均值): {mean2}")
    print(f"差異 (Img1 - Img2): {mean1 - mean2}")
    
    # 增強差異以便肉眼觀察 (將差異值放大 10 倍，不然 32 的差異在黑色背景下很難看清)
    diff_amplified = diff * 10
    
    # 儲存差異圖
    cv2.imwrite(output_diff_path, diff_amplified)
    print(f"\n✅ 已生成差異圖: {output_diff_path}")
    print("請打開這張圖：")
    print("- 如果整張圖是均勻的灰色/彩色雜訊：代表是 '亮度/Normalization' 縮放問題。")
    print("- 如果看得到原本圖片的輪廓（鬼影）：代表是 '對齊' 或 'Resize 插值' 問題。")
    print("- 如果顏色很奇怪（例如整張偏藍）：代表 RGB/BGR 通道順序搞反了。")

if __name__ == "__main__":
    # 請替換成你的檔名
    img_a = '/home/cgvmis418/Dynamic_C3DGS/log/ours_sear_steak_ori/test/ours_25000/gt/00000.png'
    img_b = '/home/cgvmis418/GSCodec_Studio/results/sear_steak/Frame0-49/renders/compress_step14999/gt/gt_testv0_fid0000.png'
    
    visualize_difference(img_a, img_b)