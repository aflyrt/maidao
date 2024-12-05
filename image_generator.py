import torch
import os
from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from datetime import datetime
import glob
from enhanced_prompt import styles
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import controlnet_launcher as cn
# from transformers import logging

# # 设置日记级别为ERROR，以减少冗余输出
# logging.set_verbosity_error()
# logging.basicConfig(level=logging.ERROR)

# 初始模型目录
model_dir = './models/cosmic-babes'    #将cosmic-babes修改为自己下载的model名称

# 检查模型目录是否存在且可读
if not os.path.exists(model_dir):
    print(f"Directory {model_dir} does not exist.")
    exit(1)

# 图片存储目录
image_folder = 'generated_images'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# 初始化管道
pipeline = StableDiffusionPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    local_files_only=True,
    safety_checker=None,
    ).to("cuda")

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

def generate_image(prompt):
    """Generate an image based on the provided prompt."""
    negative_prompt = "ugly, deformed, disfigured, bad anatomy, blurry, extra arms, extra fingers, extra limbs, extra legs, fused fingers, missing arms, missing legs, mutated hands, poorly drawn hands"
    image = pipeline(prompt, num_inference_steps=20, negative_prompt=negative_prompt).images[0]
    return image

def save_image(image, filename):
    """Save the image with the given filename."""
    image_path = os.path.join(image_folder, filename)
    image.save(image_path)
    print(f"Image generated and saved as {image_path}")

def handle_command(command, cn_mode=False):
    """Handle special commands starting with '/'."""
    command_parts = command.split()
    if len(command_parts) == 0:
        print("Invalid command format.")
        return

    cmd = command_parts[0][1:]  # Remove leading '/'
    args = command_parts[1:]

    if cmd == 'help':
        print_help()
    elif cmd == 'reset':
        reset_image_count(args)
    elif cmd == 'switch':
        switch_model(args)
    elif cmd == 'show':
        show_ascii_image(args)
    elif cmd == 'CN':    # 新增对/CN命令的支持
        if not cn_mode:
            return True # 进入CN模式
        else:
            cn.handle_controlnet_command(command, image_folder, model_dir)
    elif cmd == 'return' and cn_mode:
        return False    # 返回到prompt模式
    else:
        print(f"Unknown command '{cmd}'.")

def print_help(cn_mode=False):
    """Print available commands and styles."""
    print("Available commands:")
    print("/help - Show this help message")
    print("/reset - Reset the image count")
    print("/switch [path] - Switch to a different model")
    print("/show --bw - Show the latest generated image as a black and white ASCII art")
    print("/show --color - Show the latest generated image as a colored ASCII art")
    if cn_mode:
        print("/CN - Generate an image using ControlNet (in CN mode)")
        print("/return - Return to prompt mode")
    else:
        print("/CN - Enter ControlNet mode")
    # print("/CN - Generate an image using ControlNet")
    print("\nAvailable styles for --style command:")
    for style in styles:
        print(f"- {style}")


def reset_image_count(args):
    """Reset the image count for the day."""
    if len(args) != 1:
        print("Usage: /reset [confirm]")
        return

    confirmation = args[0]
    if confirmation.lower() == 'yes':
        current_date = datetime.now().strftime('%Y-%m-%d')
        image_count_today = len(glob.glob(os.path.join(image_folder, f"{current_date}_*.png")))
        # image_count = 1
        cn.image_count_today = 1    # 重置计数器
        print(f"Image count for {current_date} reset.")
    else:
        print("Confirmation required. Use '/reset yes' to confirm.")

def switch_model(args):
    """Switch the model used by the pipeline."""
    if len(args) != 1:
        print("Usage: /switch [path]")
        return
    
    new_model_path = args[0]
    
    # 规范化路径
    new_model_path = os.path.normpath(new_model_path)

    # 打印路径以进行调试
    print(f"Normalized path: {new_model_path}")

    # 检查新模型路径是否存在
    if not os.path.exists(new_model_path):
        print(f"Model directory {new_model_path} does not exist.")
        return
    
    global pipeline
    global model_dir
    
    # 更新模型目录
    model_dir = new_model_path
    
    # 加载新的模型
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            local_files_only=True,
            safety_checker=None).to("cuda")
        
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        # 清除缓存
        torch.cuda.empty_cache()

        print(f"Successfully switched to model in {model_dir}")
    except Exception as e:
        print(f"Error switching to model: {e}")

def get_latest_image():
    """Get the path of the latest generated image."""
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")), key=os.path.getmtime)
    if not images:
        print("No images found.")
        return None
    return images[-1]

def resize_image(image_path, output_width, output_height):
    image = Image.open(image_path)
    image = image.resize((output_width, output_height))
    return image

def convert_to_gray(image):
    return image.convert('L')

def map_pixels_to_ascii(gray_image, ascii_chars, output_width, output_height):
    ascii_image = []

    for y in range(output_height):
        row = ""
        for x in range(output_width):
            gray_value = gray_image.getpixel((x, y))
            char_index = int(gray_value / 255 * (len(ascii_chars) - 1))
            row += ascii_chars[char_index]
        ascii_image.append(row)

    return "\n".join(ascii_image)

def image_to_bw_ascii(image_path, output_width, output_height):
    resized_image = resize_image(image_path, output_width, output_height)

    gray_image = convert_to_gray(resized_image)
    
    ascii_image = map_pixels_to_ascii(gray_image, bw_ascii_chars, output_width, output_height)

    return ascii_image

def image_to_colored_ascii(image_path, output_width, output_height):
    try:
        # 调整图片大小
        resized_image = resize_image(image_path, output_width, output_height)

        # 创建 ASCII 字符画
        ascii_image = []
        ascii_image_chars = []
        ascii_image_colors = []
        for y in range(output_height):
            row = ""
            for x in range(output_width):
                # 获取像素颜色
                color = resized_image.getpixel((x, y))

                # 计算灰度值并选择相应的 ASCII 字符
                gray_value = sum(color) // 3
                char_index = int(gray_value / 255 * (len(colored_ascii_chars) - 1))
                if char_index >= len(colored_ascii_chars):
                    char_index = len(colored_ascii_chars) - 1

                # 将 ASCII 字符和颜色添加到行中
                row += f"\033[38;2;{color[0]};{color[1]};{color[2]}m{colored_ascii_chars[char_index]}"

                # 存储字符和颜色
                ascii_image_chars.append(colored_ascii_chars[char_index])
                ascii_image_colors.append((color[0], color[1], color[2]))

            # 将行添加到 ASCII 字符画中
            ascii_image.append(row)
        
            # 添加ANSI转义码重置颜色
            ascii_image.append("\033[0m")

        return "\n".join(ascii_image), ascii_image_chars, ascii_image_colors
    except Exception as e:
        print(f"Error mapping pixels to ASCII:{e}")
        return ""

def show_ascii_image(args):
    """Show the latest generated image as ASCII art."""
    if len(args) != 1 or args[0] not in ['--bw', '--color']:
        print("Usage: /show --bw or /show --color")
        return

    image_path = get_latest_image()
    if not image_path:
        return

    output_width = 128
    output_height = 60

    if args[0] == '--bw':
        bw_ascii_image = image_to_bw_ascii(image_path, output_width, output_height)
        print(bw_ascii_image)
    elif args[0] == '--color':
        colored_ascii_image, _, _ = image_to_colored_ascii(image_path, output_width, output_height)
        print(colored_ascii_image)

bw_ascii_chars = "@%#*+=-:. "
colored_ascii_chars = ["0", "1", "2", "3", "4", "5", "6", "8", "9"]
