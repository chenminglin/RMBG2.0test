import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import os
import time
from pathlib import Path
import argparse
import warnings
import psutil
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

def get_memory_usage():
    """
    获取当前进程的内存使用情况
    
    Returns:
        str: 格式化的内存使用信息
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024  # 转换为MB
    return f"{memory_mb:.1f} MB"

def load_model(model_path=None):
    """
    加载RMBG-2.0模型
    
    Args:
        model_path: 本地模型路径，如果为None则从HuggingFace下载
    
    Returns:
        model: 加载的模型
        device: 使用的设备
    """
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    if model_path is None:
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
    
    return model, device

def get_image_files(folder_path):
    """
    获取文件夹中所有支持的图片文件
    
    Args:
        folder_path: 文件夹路径
    
    Returns:
        list: 图片文件路径列表
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return []
    
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            image_files.append(file_path)
    
    return sorted(image_files)

def remove_background_single(model, device, image_path, output_path):
    """
    对单张图片进行背景移除
    
    Args:
        model: 已加载的模型
        device: 设备
        image_path: 输入图片路径
        output_path: 输出图片路径
    
    Returns:
        bool: 是否成功
        float: 处理时间（秒）
    """
    start_time = time.time()
    
    try:
        # 图像预处理设置
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # 转换图像为模型输入格式
        input_images = transform_image(image).unsqueeze(0).to(device)
        
        # 进行预测
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
        image_with_alpha.save(output_path)
        
        processing_time = time.time() - start_time
        return True, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"处理 {image_path.name} 时出错: {e}")
        return False, processing_time

def batch_remove_background(input_folder, output_folder=None, model_path=None):
    """
    批量处理文件夹中的所有图片
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径，如果为None则在输入文件夹下创建output子文件夹
        model_path: 本地模型路径
    """
    print(f"开始批量处理文件夹: {input_folder}")
    print("=" * 50)
    
    # 设置输出文件夹
    if output_folder is None:
        output_folder = Path(input_folder) / "output_no_bg"
    else:
        output_folder = Path(output_folder)
    
    # 创建输出文件夹
    output_folder.mkdir(exist_ok=True)
    print(f"输出文件夹: {output_folder}")
    
    # 获取所有图片文件
    image_files = get_image_files(input_folder)
    if not image_files:
        print("未找到支持的图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    print("支持的格式: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp")
    print("=" * 50)
    
    # 加载模型
    model_load_start = time.time()
    model, device = load_model(model_path)
    model_load_time = time.time() - model_load_start
    print(f"模型加载时间: {model_load_time:.2f} 秒")
    print("=" * 50)
    
    # 批量处理
    total_start_time = time.time()
    successful_count = 0
    failed_count = 0
    total_processing_time = 0
    
    for i, image_path in enumerate(image_files, 1):
        memory_usage = get_memory_usage()
        print(f"[{i}/{len(image_files)}] 正在处理: {image_path.name} | 内存占用: {memory_usage}")
        
        # 生成输出文件名（保持原扩展名但改为.png以支持透明度）
        output_filename = image_path.stem + "_no_bg.png"
        output_path = output_folder / output_filename
        
        # 处理单张图片
        success, processing_time = remove_background_single(
            model, device, image_path, output_path
        )
        
        total_processing_time += processing_time
        memory_usage_after = get_memory_usage()
        
        if success:
            successful_count += 1
            print(f"  ✓ 成功 - 耗时: {processing_time:.2f} 秒 | 处理后内存: {memory_usage_after}")
        else:
            failed_count += 1
            print(f"  ✗ 失败 - 耗时: {processing_time:.2f} 秒 | 处理后内存: {memory_usage_after}")
        
        print()
    
    # 统计结果
    total_time = time.time() - total_start_time
    
    print("=" * 50)
    print("批量处理完成！")
    print("=" * 50)
    print(f"总文件数: {len(image_files)}")
    print(f"成功处理: {successful_count}")
    print(f"处理失败: {failed_count}")
    print(f"成功率: {successful_count/len(image_files)*100:.1f}%")
    print("=" * 50)
    print("时间统计:")
    print(f"模型加载时间: {model_load_time:.2f} 秒")
    print(f"图片处理总时间: {total_processing_time:.2f} 秒")
    print(f"平均每张图片处理时间: {total_processing_time/len(image_files):.2f} 秒")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"输出文件夹: {output_folder}")

def main():
    """
    主函数 - 命令行接口
    """
    parser = argparse.ArgumentParser(
        description='使用RMBG-2.0模型批量移除图片背景',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  python batch_rmbg.py -i /path/to/input/folder -o /path/to/output/folder
  python batch_rmbg.py -i ./images -o ./results -m /path/to/model
  python batch_rmbg.py -i ./images  # 输出到 ./images/output_no_bg
        '''
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入图片文件夹路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出文件夹路径（可选，默认为输入文件夹下的output_no_bg子文件夹）'
    )
    
    parser.add_argument(
        '-m', '--model',
        help='本地模型路径（可选，默认使用预设的本地模型路径）'
    )
    
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output
    model_path = args.model
    
    print(f"使用命令行参数:")
    print(f"  输入文件夹: {input_folder}")
    print(f"  输出文件夹: {output_folder if output_folder else '自动生成'}")
    print(f"  模型路径: {model_path if model_path else '使用默认路径'}")
    print("=" * 50)
    
    # 检查输入文件夹
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在")
        return
    
    # 执行批量处理
    batch_remove_background(
        input_folder=input_folder,
        output_folder=output_folder,
        model_path=model_path
    )

if __name__ == "__main__":
    main()