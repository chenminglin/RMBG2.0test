import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import time
from pathlib import Path
import argparse
import warnings
import psutil
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

# 设置为ERROR级别，只显示错误信息
onnxruntime.set_default_logger_severity(3)

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

def load_onnx_model(model_path):
    """
    加载 ONNX 模型

    Args:
        model_path: ONNX 模型文件路径

    Returns:
        session: ONNX InferenceSession
        input_name: 模型输入名称
        output_name: 模型输出名称
    """
    print(f"正在加载 ONNX 模型: {model_path}")
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name # 假设模型只有一个输出，或者我们关心的是第一个输出
    # 如果有多个输出，可能需要更复杂的逻辑来确定哪个是主要的分割掩码
    # 例如，可以检查 session.get_outputs() 返回的列表
    # print("Model Outputs:", [output.name for output in session.get_outputs()])
    # 对于 RMBG-2.0 ONNX，通常输出是 'mask'
    # 确保这里的 output_name 是正确的，如果不是 'mask'，可能需要调整
    # 经过检查，Hugging Face 上的 RMBG-2.0 ONNX 模型的输出名通常是 'mask' 或类似的
    # 如果不确定，可以先打印出来看看
    # print(f"Model Input Name: {input_name}")
    # print(f"Model Output Name: {output_name}")
    return session, input_name, output_name

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

def remove_background_single_onnx(session, input_name, output_name, image_path, output_path):
    """
    对单张图片进行背景移除 (ONNX)

    Args:
        session: ONNX InferenceSession
        input_name: 模型输入名称
        output_name: 模型输出名称
        image_path: 输入图片路径
        output_path: 输出图片路径

    Returns:
        bool: 是否成功
        float: 处理时间（秒）
    """
    start_time = time.time()

    try:
        # 图像预处理设置 (与PyTorch版本保持一致)
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
        input_tensor = transform_image(image).unsqueeze(0).numpy()

        # 进行预测
        ort_inputs = {input_name: input_tensor}
        ort_outs = session.run([output_name], ort_inputs)
        pred_tensor = ort_outs[0] # 获取输出张量

        # 处理预测结果 (假设输出是 sigmoid 后的概率图)
        # ONNX 模型输出的形状可能是 (1, 1, H, W) 或 (1, H, W)
        # 我们需要将其转换为 (H, W)
        pred = np.squeeze(pred_tensor) # 移除批次和通道维度 (如果通道为1)
        if pred.ndim == 3 and pred.shape[0] == 1: # (1, H, W)
            pred = pred.squeeze(0)
        elif pred.ndim != 2: # (H, W)
            raise ValueError(f"Unexpected prediction shape: {pred.shape}")

        # 将概率图转换为 PIL Image
        # 确保 pred 的值在 [0, 1] 范围内，如果不是，可能需要 sigmoid
        # RMBG-2.0 ONNX 模型通常已经包含了 sigmoid
        pred_pil = Image.fromarray((pred * 255).astype(np.uint8), mode='L')

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
        print(f"处理 {image_path.name} 时出错 (ONNX): {e}")
        import traceback
        traceback.print_exc()
        return False, processing_time

def batch_remove_background_onnx(input_folder, output_folder=None, onnx_model_path=None):
    """
    批量处理文件夹中的所有图片 (ONNX)

    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径，如果为None则在输入文件夹下创建output_onnx子文件夹
        onnx_model_path: ONNX 模型文件路径
    """
    print(f"开始使用 ONNX 模型批量处理文件夹: {input_folder}")
    print("=" * 50)

    # 设置输出文件夹
    if output_folder is None:
        output_folder = Path(input_folder) / "output_no_bg_onnx"
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
    if onnx_model_path is None:
        # 默认使用项目中的 onnx/model.onnx
        # 获取当前脚本所在的目录
        current_script_dir = Path(__file__).resolve().parent
        onnx_model_path = current_script_dir / "onnx" / "model.onnx"
        if not onnx_model_path.exists():
            print(f"错误: 默认 ONNX 模型路径 {onnx_model_path} 不存在。请提供正确的 onnx_model_path 参数。")
            # 尝试其他常见的 ONNX 模型名称
            potential_models = ["model_fp16.onnx", "model_quantized.onnx"]
            for pm_name in potential_models:
                pm_path = current_script_dir / "onnx" / pm_name
                if pm_path.exists():
                    onnx_model_path = pm_path
                    print(f"找到备选模型: {onnx_model_path}")
                    break
            else:
                 print(f"错误: 在 {current_script_dir / 'onnx'} 目录下未找到可用的 ONNX 模型。")
                 return

    model_load_start = time.time()
    try:
        session, input_name, output_name = load_onnx_model(str(onnx_model_path))
    except Exception as e:
        print(f"加载 ONNX 模型失败: {e}")
        return
    model_load_time = time.time() - model_load_start
    print(f"ONNX 模型加载时间: {model_load_time:.2f} 秒")
    print("=" * 50)

    # 批量处理
    total_start_time = time.time()
    successful_count = 0
    failed_count = 0
    total_processing_time = 0

    for i, image_path in enumerate(image_files, 1):
        memory_usage = get_memory_usage()
        print(f"[{i}/{len(image_files)}] 正在处理 (ONNX): {image_path.name} | 内存占用: {memory_usage}")

        # 生成输出文件名（保持原扩展名但改为.png以支持透明度）
        output_filename = image_path.stem + "_no_bg.png"
        output_path = output_folder / output_filename

        # 处理单张图片
        success, processing_time = remove_background_single_onnx(
            session, input_name, output_name, image_path, output_path
        )

        total_processing_time += processing_time
        memory_usage_after = get_memory_usage()

        if success:
            successful_count += 1
            print(f"  ✓ 成功 (ONNX) - 耗时: {processing_time:.2f} 秒 | 处理后内存: {memory_usage_after}")
        else:
            failed_count += 1
            print(f"  ✗ 失败 (ONNX) - 耗时: {processing_time:.2f} 秒 | 处理后内存: {memory_usage_after}")

        print()

    # 统计结果
    total_time = time.time() - total_start_time

    print("=" * 50)
    print("ONNX 批量处理完成！")
    print("=" * 50)
    print(f"总文件数: {len(image_files)}")
    print(f"成功处理: {successful_count}")
    print(f"处理失败: {failed_count}")
    if len(image_files) > 0:
        print(f"成功率: {successful_count/len(image_files)*100:.1f}%")
        print(f"平均每张图片处理时间: {total_processing_time/len(image_files):.2f} 秒")
    print("=" * 50)
    print("时间统计:")
    print(f"ONNX 模型加载时间: {model_load_time:.2f} 秒")
    print(f"图片处理总时间: {total_processing_time:.2f} 秒")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"输出文件夹: {output_folder}")

def main():
    """
    主函数 - 命令行接口
    """
    parser = argparse.ArgumentParser(
        description='使用ONNX模型批量移除图片背景',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  python batch_rmbg_onnx.py -i /path/to/input/folder -o /path/to/output/folder
  python batch_rmbg_onnx.py -i ./images -o ./results -m ./model.onnx
  python batch_rmbg_onnx.py -i ./images  # 输出到 ./images/output_no_bg_onnx
        '''
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入图片文件夹路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出文件夹路径（可选，默认为输入文件夹下的output_no_bg_onnx子文件夹）'
    )
    
    parser.add_argument(
        '-m', '--model',
        help='ONNX模型文件路径（可选，默认在./onnx/目录下自动查找）'
    )
    
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output
    onnx_model_path = args.model
    
    print(f"使用命令行参数:")
    print(f"  输入文件夹: {input_folder}")
    print(f"  输出文件夹: {output_folder if output_folder else '自动生成'}")
    print(f"  ONNX模型: {onnx_model_path if onnx_model_path else '自动查找'}")
    print("=" * 50)
    
    # 检查输入文件夹
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在")
        return
    
    # 执行批量处理
    batch_remove_background_onnx(
        input_folder=input_folder,
        output_folder=output_folder,
        onnx_model_path=onnx_model_path
    )

if __name__ == "__main__":
    main()