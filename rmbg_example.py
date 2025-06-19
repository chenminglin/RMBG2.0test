import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import matplotlib.pyplot as plt
import os

def remove_background(input_image_path, output_image_path):
    """
    使用RMBG-2.0模型移除图片背景
    
    Args:
        input_image_path: 输入图片路径
        output_image_path: 输出图片路径
    """
    
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载RMBG-2.0模型...")
    model_path = "/Users/chenminglin/hf-download/RMBG-2.0"  # 本地模型路径
    model = AutoModelForImageSegmentation.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True  # 强制只使用本地文件
    )
    
    # 设置模型精度和设备
    if device == 'cuda':
        torch.set_float32_matmul_precision('high')
    model.to(device)
    model.eval()
    
    # 图像预处理设置
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载和预处理图像
    print(f"正在处理图像: {input_image_path}")
    try:
        image = Image.open(input_image_path).convert('RGB')
        original_size = image.size
        print(f"原始图像尺寸: {original_size}")
    except Exception as e:
        print(f"无法加载图像: {e}")
        return False
    
    # 转换图像为模型输入格式
    input_images = transform_image(image).unsqueeze(0).to(device)
    
    # 进行预测
    print("正在进行背景移除...")
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    
    # 处理预测结果
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    
    # 将mask调整回原始图像尺寸
    mask = pred_pil.resize(original_size, Image.LANCZOS)
    
    # 应用alpha通道（透明背景）
    image_with_alpha = image.copy()
    image_with_alpha.putalpha(mask)
    
    # 保存结果
    image_with_alpha.save(output_image_path)
    print(f"背景移除完成，结果保存至: {output_image_path}")
    
    return True


def main():
    """
    主函数 - 命令行接口
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='使用RMBG-2.0模型移除单张图片背景',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  python rmbg_example.py -i input.jpg -o output.png
  python rmbg_example.py -i /path/to/image.jpg -o /path/to/result.png
  python rmbg_example.py -i image.jpg  # 输出为 image_no_bg.png
        '''
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入图片文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出图片文件路径（可选，默认为输入文件名_no_bg.png）'
    )
    
    args = parser.parse_args()
    
    input_image = args.input
    
    # 如果没有指定输出路径，自动生成
    if args.output:
        output_image = args.output
    else:
        # 从输入文件名生成输出文件名
        input_path = os.path.splitext(input_image)
        output_image = input_path[0] + "_no_bg.png"
    
    print(f"使用命令行参数:")
    print(f"  输入图片: {input_image}")
    print(f"  输出图片: {output_image}")
    print("=" * 50)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_image):
        print(f"错误: 输入图片 '{input_image}' 不存在")
        return
    
    # 执行背景移除
    success = remove_background(input_image, output_image)
    
    if success:
        print("\n处理完成！")
        print(f"- 原始图片: {input_image}")
        print(f"- 背景移除结果: {output_image}")
    else:
        print("背景移除失败")

if __name__ == "__main__":
    main()