import os 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/.cache/huggingface'
os.environ['HF_HOME'] = 'D:/.cache/huggingface'

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import diffusers
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector
import torch
import numpy as np
import cv2
from datetime import datetime
import argparse
import glob
# from enhanced_prompt import styles, optimize_prompt  # 导入 styles 和 optimize_prompt


# 全局变量，用于跟踪当天已生成的图片数量
# image_count_today = 0
current_date = datetime.now().strftime('%Y-%m-%d')

def preprocess_openpose(image_path):
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    image = load_image(image_path)
    # print(f"Preprocessing with OpenPose: {image_path}")
    # resized_image = resize_image(image)
    preprocessed_image = openpose(image)
    # preprocessed_image.show()  # 显示预处理后的图像以进行验证
    return preprocessed_image

def preprocess_canny(image_path):
    image = load_image(image_path)
    # resized_image = resize_image(image)
    image_np = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image_canny = cv2.Canny(image_np, low_threshold, high_threshold)
    image_canny = image_canny[:, :, None]
    image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
    image_canny_pil = Image.fromarray(image_canny)
    # print(f"Preprocessing with Canny: {image_path}")
    # image_canny_pil.show()  # 显示预处理后的图像以进行验证
    return image_canny_pil

def preprocess_depth(image_path):
    marigold = diffusers.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
    ).to("cuda")
    image = load_image(image_path)
    # resized_image = resize_image(image)
    depth = marigold(image)
    vis = marigold.image_processor.visualize_depth(depth.prediction)
    # print(f"Preprocessing with Depth: {image_path}")
    # vis[0].show()  # 显示预处理后的图像以进行验证
    return vis[0]

def generate_image(prompt, image, model_path, controlnet_model_name, num_inference_steps=20):
    controlnet = ControlNetModel.from_pretrained(controlnet_model_name, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_path, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    # 打印提示词和控制网络模型名称以进行验证
    print(f"Generating image with prompt: {prompt} and controlnet model: {controlnet_model_name}")
    negative_prompt = "ugly, deformed, disfigured, bad anatomy, blurry, extra arms, extra fingers, extra limbs, extra legs, fused fingers, missing arms, missing legs, mutated hands, poorly drawn hands"
    generated_image = pipe(prompt, image, height=512, width=512, num_inference_steps=num_inference_steps, negative_prompt=negative_prompt).images[0]
    return generated_image

def save_image(image, image_folder, current_date, image_count):
    """Save the image with the given filename."""
    image_filename = f"{current_date}_generated_image_{image_count:04d}.png"
    image_path = os.path.join(image_folder, image_filename)
    image.save(image_path)
    print(f"Image generated and saved as {image_path}")

def handle_controlnet_input(command, image_folder, model_dir):
    # global image_count_today
    # global current_date

    parser = argparse.ArgumentParser(description="Generate images using ControlNet.")
    parser.add_argument("prompt", type=str, help="The text prompt for image generation.")
    parser.add_argument("--openpose", action='store_true', help="Use OpenPose preprocessor.")
    parser.add_argument("--canny", action='store_true', help="Use Canny edge detection preprocessor.")
    parser.add_argument("--depth", action='store_true', help="Use depth estimation preprocessor.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    # parser.add_argument("--style", type=str, choices=list(styles.keys()), default="cinematic", help="Style of the image generation.")
    
    # 使用 shlex.split 来处理命令字符串，确保参数正确分割
    import shlex
    args = parser.parse_args(shlex.split(command))
    
    if sum([args.openpose, args.canny, args.depth]) != 1:
        raise ValueError("Exactly one preprocessor (--openpose, --canny, --depth) must be specified.")
    
    prompt = args.prompt
    # # 优化提示词
    # enhanced_prompt = optimize_prompt(args.prompt, args.style)
    
    if args.openpose:
        preprocessed_image = preprocess_openpose(args.image_path)
        controlnet_model_name = "lllyasviel/sd-controlnet-openpose"
    elif args.canny:
        preprocessed_image = preprocess_canny(args.image_path)
        controlnet_model_name = "lllyasviel/sd-controlnet-canny"
    elif args.depth:
        preprocessed_image = preprocess_depth(args.image_path)
        controlnet_model_name = "lllyasviel/sd-controlnet-depth"
    
    generated_image = generate_image(prompt, preprocessed_image, model_dir, controlnet_model_name)
    image_count_today = len(glob.glob(os.path.join(image_folder, f"{current_date}_*.png")))
    image_count = image_count_today + 1
    save_image(generated_image, image_folder, current_date, image_count)
    image_count += 1



