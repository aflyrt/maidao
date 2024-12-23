# maidao 

<pre>
      __  ___         __  _                                  
   /  |/  /________/ / (_)___  __  ___________  ___  __  __
  / /|_/ / ___/ __  / / / __ \/ / / / ___/ __ \/ _ \/ / / /
 / /  / / /__/ /_/ / / / /_/ / /_/ / /  / / / /  __/ /_/ / 
/_/  /_/\___/\__,_/_/ /\____/\__,_/_/  /_/ /_/\___/\__, /  
                 /___/                            /____/   
</pre>

## 介绍
**“麦道”（Mcdjourney）** 是一款通过powershell或cmd交互实现文生图的应用，支持StableDiffusion和ControlNet，旨在帮助用户使用计算机的本地硬件资源使用文生图服务。
“麦道”（Mcdjourney）的名字将McDonald's和Midjourney结合，是对先进的文生图应用Midjourney的一种致敬，但除名称外与二者均无任何关联。
## 硬件要求
* NVIDIA GeForce RTX 3060显卡（6GB显存）及以上
## 安装
* 确保安装到`D:\`盘。`git clone https://github.com/aflyrt/maidao.git "D:\"`（否则需要修改程序才能正常使用）。
* 创建虚拟环境并安装依赖包。
```
cd D:\maidao
conda create -name Mcdjourney python=3.10
conda activate Mcdjourney
pip install -r requirements.txt
```
* 下载必要的模型。运行`models`和`MagicPrompt-Stable-Diffusion`文件夹中的`snapshot_download.ipynb`下载必要的模型。
## 使用
打开powershell或cmd，确保在`\Mcdjourney`目录下，启动`main.py`，`python main.py`，等待加载完成出现`/prompt`提示符即可使用。
### 1. 基本操作
Mcdjourney支持/prompt（提示词）和/CN（ControlNet）两种模式：
* 默认使用/prompt模式，在prompt模式下输入 `PROMPT --style STYLE` 进行文生图，style参数为非必须，如不输入默认为cinematic风格。

  例如：
  > `a young man driving a roller coaster in Disneyland`
  > 
  > `a young man driving roller coaster in Disneyland --style lineart`
* 在/prompt模式下，输入`/CN`进入/CN模式，在/CN模式下输入`/return`返回/prompt模式，在任意模式下输入`exit`退出程序。
* 在/CN模式下，输入 `"PROMPT" --<预处理器> --image_path IMAGE_PATH `进行文生图，/CN模式暂不支持输入style参数。请注意，在/CN模式下，提示词必须**处于双引号（""）中**，否则无法识别。

  例如：
  > `"yoga on spacecraft" --canny --image_path "D:\maidao\yoga.jpg"`
  > 
  >` "a youngman playing baseball game on Mars" --openpose --image_path "D:\maidao\baseball_player.jpg"`
### 2. 特殊指令
* 在/prompt模式下，输入/help指令可查看支持的命令和风格参数。
  |Command|Description|
  |---|---|
  |/help|Show help message|
  |/switch [path]|Switch to different model|
  |/show --bw|Print the latest generated image as a black and white ASCII art|
  |/show --color|Print the latest generated image as a colored ASCII art|
  |/CN|Generate images using ContronNet|
  |/return|Return to prompt mode(in CN mode)|
### 3. 查看图像
无论使用/prompt模式还是/CN模式，所有生成的图像均存储在generated_images文件夹下，命名方式为`yyyy-mm-dd_generated_image_count.png`。
## 感谢
本项目使用了以下作者的开源代码和库：
* [huggingface/diffusers](https://github.com/huggingface/diffusers)
* [huggingface/controlnet_aux](https://github.com/huggingface/controlnet_aux)
* [lllyasviel/controlnet](https://github.com/lllyasviel/ControlNet)
