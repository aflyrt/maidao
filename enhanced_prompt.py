import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/.cache/huggingface'
os.environ['HF_HOME'] = 'D:/.cache/huggingface'
import torch
from transformers import GenerationConfig, GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor, LogitsProcessorList

styles = {
    "cinematic": "cinematic film still of {prompt}, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain",
    "anime": "anime artwork of {prompt}, anime style, key visual, vibrant, studio anime, highly detailed",
    "photographic": "cinematic photo of {prompt}, 35mm photograph, film, professional, 4k, highly detailed",
    "comic": "comic of {prompt}, graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "lineart": "line art drawing {prompt}, professional, sleek, modern, minimalist, graphic, line art, vector graphics",
    "pixelart": " pixel-art {prompt}, low-res, blocky, pixel art style, 8-bit graphics",
}

words = [
    "aesthetic", "astonishing", "beautiful", "breathtaking", "composition", "contrasted", "epic", "moody", "enhanced",
    "exceptional", "fascinating", "flawless", "glamorous", "glorious", "illumination", "impressive", "improved",
    "inspirational", "magnificent", "majestic", "hyperrealistic", "smooth", "sharp", "focus", "stunning", "detailed",
    "intricate", "dramatic", "high", "quality", "perfect", "light", "ultra", "highly", "radiant", "satisfying",
    "soothing", "sophisticated", "stylish", "sublime", "terrific", "touching", "timeless", "wonderful", "unbelievable",
    "elegant", "awesome", "amazing", "dynamic", "trendy",
]

word_pairs = ["highly detailed", "high quality", "enhanced quality", "perfect composition", "dynamic light"]

def find_and_order_pairs(s, pairs):
    words = s.split()
    found_pairs = []
    for pair in pairs:
        pair_words = pair.split()
        if pair_words[0] in words and pair_words[1] in words:
            found_pairs.append(pair)
            words.remove(pair_words[0])
            words.remove(pair_words[1])

    for word in words[:]:
        for pair in pairs:
            if word in pair.split():
                words.remove(word)
                break

    ordered_pairs = ", ".join(found_pairs)
    remaining_s = ", ".join(words)
    
    # 打印结果
    print("Ordered Pairs:", ordered_pairs)
    print("Remaining Words:", remaining_s)
    return ordered_pairs, remaining_s

def optimize_prompt(prompt, style):
    global words, word_pairs #明确引用全局变量
    
    prompt = styles[style].format(prompt=prompt)
    # 打印调试信息
    print(f"Initial Prompt: {prompt}")

    MagicPrompt = "./MagicPrompt-Stable-Diffusion"
    tokenizer = GPT2Tokenizer.from_pretrained(MagicPrompt)
    model = GPT2LMHeadModel.from_pretrained(MagicPrompt, torch_dtype=torch.float16).to("cuda")
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    token_count = inputs["input_ids"].shape[1]
    max_new_tokens = 50 - token_count

    generation_config = GenerationConfig(
        penalty_alpha=0.7,
        top_k=50,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.eos_token_id,
        pad_token=model.config.pad_token_id,
        do_sample=True,
    )

    class CustomLogitsProcessor(LogitsProcessor):
        def __init__(self, bias):
            super().__init__()
            self.bias = bias

        def __call__(self, input_ids, scores):
            if len(input_ids.shape) == 2:
                last_token_id = input_ids[0, -1]
                self.bias[last_token_id] = -1e10
            return scores + self.bias

    word_ids = [tokenizer.encode(word, add_prefix_space=True)[0] for word in words]
    bias = torch.full((tokenizer.vocab_size,), -float("Inf")).to("cuda")
    bias[word_ids] = 0
    processor = CustomLogitsProcessor(bias)
    processor_list = LogitsProcessorList([processor])

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
            logits_processor=processor_list,
        )

    output_tokens = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]
    input_part, generated_part = output_tokens[0][: len(prompt)], output_tokens[0][len(prompt) :]
    pairs, words = find_and_order_pairs(generated_part, word_pairs)
    print(f"Generated Part: {generated_part}")
    formatted_generated_part = pairs + ", " + words
    enhanced_prompt = input_part + ", " + formatted_generated_part
    print(f"Enhanced Prompt: {enhanced_prompt}")
    return enhanced_prompt