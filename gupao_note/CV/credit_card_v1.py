import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

def cv_show(name, img):
    """显示图像的辅助函数"""
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_image(img):
    """预处理图像：灰度化、高斯模糊、边缘检测"""
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 边缘检测
    edged = cv2.Canny(blur, 75, 200)
    return gray, edged

def find_contours(edged):
    """寻找轮廓并按面积排序"""
    # 寻找轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 按轮廓面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours

def extract_card_number_region(contours, img):
    """提取信用卡数字区域"""
    for c in contours:
        # 计算轮廓周长
        peri = cv2.arcLength(c, True)
        # 多边形逼近
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # 如果是四边形，可能是数字区域
        if len(approx) == 4:
            number_region = approx
            break
    
    # 绘制轮廓
    cv2.drawContours(img, [number_region], -1, (0, 255, 0), 2)
    
    # 获取数字区域的坐标
    (x, y, w, h) = cv2.boundingRect(number_region)
    # 提取数字区域
    number_roi = img[y:y+h, x:x+w]
    
    return number_roi

def load_digit_templates(template_path):
    """加载数字模板并处理"""
    # 读取模板图像
    template = cv2.imread(template_path)
    if template is None:
        raise FileNotFoundError(f"无法找到模板文件: {template_path}\n请确保digits_template.png文件位于{os.path.dirname(template_path)}目录下")
    
    # 转换为灰度图并二值化
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, template_thresh = cv2.threshold(template_gray, 10, 255, cv2.THRESH_BINARY_INV)
    
    # 寻找模板中的轮廓
    template_contours, _ = cv2.findContours(template_thresh.copy(), 
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 按轮廓面积排序
    template_contours = sorted(template_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    # 存储每个数字的模板
    digit_templates = {}
    
    # 遍历轮廓，提取每个数字的模板
    for c in template_contours:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = template_thresh[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 88))  # 统一大小
        digit_templates[len(digit_templates)] = roi
    
    return digit_templates

def recognize_digits(number_roi, digit_templates):
    """识别信用卡上的数字"""
    # 预处理数字区域
    roi_gray = cv2.cvtColor(number_roi, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(roi_gray, 10, 255, cv2.THRESH_BINARY_INV)
    
    # 寻找数字区域中的轮廓
    roi_contours, _ = cv2.findContours(roi_thresh.copy(), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 存储数字轮廓
    digit_contours = []
    
    # 筛选可能是数字的轮廓
    for c in roi_contours:
        (x, y, w, h) = cv2.boundingRect(c)
        # 根据宽高比判断是否为数字
        if 20 <= w <= 50 and 80 <= h <= 120:
            digit_contours.append((x, y, w, h))
    
    # 按x坐标排序，确保从左到右识别
    digit_contours = sorted(digit_contours, key=lambda x: x[0])
    
    # 存储识别结果
    recognized_digits = []
    
    # 对每个数字进行识别
    for (x, y, w, h) in digit_contours:
        # 提取单个数字
        digit_roi = roi_thresh[y:y+h, x:x+w]
        digit_roi = cv2.resize(digit_roi, (57, 88))  # 调整为与模板相同大小
        
        # 初始化匹配分数
        scores = []
        
        # 与每个模板进行匹配
        for digit, template in digit_templates.items():
            # 模板匹配
            result = cv2.matchTemplate(digit_roi, template, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        
        # 找到最佳匹配
        recognized_digit = np.argmax(scores)
        recognized_digits.append(str(recognized_digit))
        
        # 在图像上绘制识别结果
        cv2.rectangle(number_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(number_roi, str(recognized_digit), (x - 10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    
    return recognized_digits, number_roi

def main():
    # 构建完整的文件路径
    card_img_path = os.path.join(script_dir, "credit_card.jpg")
    template_path = os.path.join(script_dir, "digits_template.png")
    
    # 读取信用卡图像
    card_img = cv2.imread(card_img_path)
    if card_img is None:
        print(f"无法找到信用卡图像文件: {card_img_path}")
        return
    
    # 预处理图像
    gray, edged = preprocess_image(card_img)
    
    # 寻找轮廓
    contours = find_contours(edged)
    
    try:
        # 提取数字区域
        number_roi = extract_card_number_region(contours, card_img.copy())
        
        # 加载数字模板
        digit_templates = load_digit_templates(template_path)
        
        # 识别数字
        recognized_digits, result_img = recognize_digits(number_roi, digit_templates)
        
        # 显示结果
        print(f"识别结果: {''.join(recognized_digits)}")
        
        # 显示处理结果
        plt.figure(figsize=(15, 10))
        plt.subplot(221), plt.imshow(cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)), plt.title("原始图像")
        plt.subplot(222), plt.imshow(edged, cmap="gray"), plt.title("边缘检测")
        plt.subplot(223), plt.imshow(cv2.cvtColor(number_roi, cv2.COLOR_BGR2RGB)), plt.title("数字区域")
        plt.subplot(224), plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)), plt.title("识别结果")
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()