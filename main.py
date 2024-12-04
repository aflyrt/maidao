import os 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/.cache/huggingface'
os.environ['HF_HOME'] = 'D:/.cache/huggingface'

import pyfiglet
import image_generator as ig
from datetime import datetime
import glob
from enhanced_prompt import optimize_prompt, styles, words, word_pairs
import controlnet_launcher as cn

# 打印ASCII艺术文字“Mcdjourney”
acsii_art = pyfiglet.figlet_format("Mcdjourney", font="slant", justify="center", width=200)
print(acsii_art)

# 默认风格
default_style = "cinematic"

def main_loop():
    global image_count_today
    cn_mode = False # 控制是否处于CN模式

    while True:

        # 获取当前日期
        current_date = datetime.now().strftime('%Y-%m-%d')

        # 查找当天已生成的图片数量
        image_count_today = len(glob.glob(os.path.join(ig.image_folder, f"{current_date}_*.png")))
        image_count = image_count_today + 1  # 从当天的计数开始
        
        # 从命令行接收用户输入
        # command = input("/prompt:")
        command = input("/CN:" if cn_mode else "/prompt:")
        
        # 命令解析
        if command.startswith('/'):
            if command == '/CN':
                cn_mode = True
            elif command == '/return' and cn_mode:
                cn_mode = False
            else:
                if cn_mode:
                    try:
                        # 调用 controlnet_launcher 处理 CN 模式下的命令
                        cn.handle_controlnet_input(command, ig.image_folder, ig.model_dir)
                    except Exception as e:
                        print(f"Error: {e}")
                else:
                    # 调用 image_generator 处理普通模式下的命令
                    should_switch_to_cn = ig.handle_command(command)
                    if should_switch_to_cn:
                        cn_mode = True
        elif command.lower() == 'exit':
            print("Goodbye! (,,・ω・,,)")
            break
        else:
            # 解析用户输入的提示词和样式
            parts = command.rsplit(' --style', 1)
            if len(parts) == 2:
                prompt = parts[0].strip()   # 去除提示词前后的空白字符
                style = parts[1].strip()    # 去除样式前后的空白字符
            else:
                prompt = command.strip()
                style = default_style

            if style not in styles:
                print(f"Invalid style '{style}'. Available styles: {list(styles.keys())}")
                continue
            
            if cn_mode:
                try:
                    # 调用 controlnet_launcher 处理 CN 模式下的命令
                    cn.handle_controlnet_input(command, ig.image_folder, ig.model_dir)
                except Exception as e:
                    print(f"Error: {e}")
            else:
            
                # 优化提示词
                enhanced_prompt = optimize_prompt(prompt, style)
                
                # 生成图片
                image = ig.generate_image(enhanced_prompt)
            

                # 构建基于日期和计数的文件名
                image_filename = f"{current_date}_generated_image_{image_count:04d}.png"
                ig.save_image(image, image_filename)
                    
                # 更新计数器
                image_count += 1

# 主程序入口
if __name__ == '__main__':
    main_loop()