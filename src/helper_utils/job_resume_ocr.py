import os
import argparse
import tempfile
from pdf2image import convert_from_path
import easyocr

def create_directory(directory):
    """创建目录（如果不存在）"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def pdf_to_images(pdf_path, temp_dir):
    """
    将PDF文件转换为图片
    
    参数:
        pdf_path: PDF文件路径
        temp_dir: 临时目录，用于存储转换后的图片
        
    返回:
        图片文件路径列表，如果出错则返回空列表
    """
    try:
        print(f"正在将PDF转换为图片: {pdf_path}")
        # 将PDF转换为图片，使用较高的dpi以获得更好的识别效果
        images = convert_from_path(pdf_path, output_folder=temp_dir, dpi=300)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f"page_{i+1}.png")
            image.save(image_path, "PNG")
            image_paths.append(image_path)
        print(f"成功将PDF转换为 {len(image_paths)} 张图片")
        return image_paths
    except Exception as e:
        print(f"转换PDF到图片时出错: {e}")
        return []

def ocr_images(image_paths, lang=['ch_sim', 'en']):
    """
    使用EasyOCR识别图片中的文本
    
    参数:
        image_paths: 图片文件路径列表
        lang: 识别语言，默认为中文简体和英文
        
    返回:
        识别到的文本，如果出错则返回空字符串
    """
    try:
        print(f"初始化OCR阅读器，识别语言: {lang}")
        # 初始化EasyOCR阅读器
        reader = easyocr.Reader(lang)
        
        full_text = []
        for i, image_path in enumerate(image_paths):
            print(f"正在识别第 {i+1}/{len(image_paths)} 页")
            # 读取图片并识别文本
            result = reader.readtext(image_path)
            
            # 提取识别结果中的文本
            page_text = []
            for detection in result:
                page_text.append(detection[1])
            
            full_text.append('\n'.join(page_text))
        
        return '\n\n'.join(full_text)  # 页与页之间用两个换行分隔
    except Exception as e:
        print(f"OCR识别时出错: {e}")
        return ""

def save_text(text, output_path):
    """
    将文本保存到TXT文件
    
    参数:
        text: 要保存的文本
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"已保存识别结果到: {output_path}")
    except Exception as e:
        print(f"保存文本时出错: {e}")

def process_pdf(pdf_path, output_dir):
    """
    处理单个PDF文件：转换为图片 -> OCR识别 -> 保存为TXT
    
    参数:
        pdf_path: PDF文件路径
        output_dir: 输出目录
    """
    print(f"\n===== 开始处理: {pdf_path} =====")
    
    # 创建临时目录存储PDF转换的图片
    with tempfile.TemporaryDirectory() as temp_dir:
        # 转换PDF为图片
        image_paths = pdf_to_images(pdf_path, temp_dir)
        if not image_paths:
            print(f"处理 {pdf_path} 失败: 无法转换为图片")
            return
        
        # 进行OCR识别
        text = ocr_images(image_paths)
        if not text:
            print(f"处理 {pdf_path} 失败: OCR识别无结果")
            return
        
        # 构建输出文件路径
        pdf_filename = os.path.basename(pdf_path)
        txt_filename = os.path.splitext(pdf_filename)[0] + '.txt'
        output_path = os.path.join(output_dir, txt_filename)
        
        # 保存识别结果
        save_text(text, output_path)
    
    print(f"===== {pdf_path} 处理完成 =====")

def process_directory(input_dir, output_dir):
    """
    处理目录下的所有PDF文件
    
    参数:
        input_dir: 输入目录，包含PDF文件
        output_dir: 输出目录，用于保存TXT文件
    """
    # 确保输出目录存在
    create_directory(output_dir)
    
    # 统计目录中的PDF文件数量
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    print(f"在 {input_dir} 目录中找到 {len(pdf_files)} 个PDF文件")
    
    # 遍历输入目录中的所有PDF文件
    for i, filename in enumerate(pdf_files):
        print(f"\n处理进度: {i+1}/{len(pdf_files)}")
        pdf_path = os.path.join(input_dir, filename)
        process_pdf(pdf_path, output_dir)

def main():
    """主函数：解析命令行参数并开始处理PDF文件"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='使用EasyOCR将目录下的PDF文件转换为TXT文件',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input', required=True, help='PDF文件所在的输入目录')
    parser.add_argument('--output', required=True, help='TXT文件保存的输出目录')
    
    # 添加使用示例
    parser.epilog = """使用示例:
  python pdf_to_txt.py --input ./pdf_files --output ./txt_results
  上述命令会将./pdf_files目录下的所有PDF文件转换为TXT文件，并保存到./txt_results目录"""
    
    args = parser.parse_args()
    
    # 验证输入目录是否存在
    if not os.path.isdir(args.input):
        print(f"错误: 输入目录 '{args.input}' 不存在")
        return
    
    # 处理目录下的所有PDF文件
    process_directory(args.input, args.output)
    print("\n所有PDF文件处理完毕")

if __name__ == "__main__":
    main()