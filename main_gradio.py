import logging
import os

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from tools.logging_utils import log_set


def get_model(model_path: str = "X-D-Lab/MindChat-Qwen-7B"):
    logging.info(f"Starting Loading {os.path.basename(model_path)}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True,
                                                 fp16=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    logging.info(f"Loading {os.path.basename(model_path)} finished")
    return tokenizer, model


def common_demo(model_path: str):
    tokenizer, model = get_model(model_path=model_path)
    history = []

    while True:
        history = history[-20:]
        message = input("User: ")
        response, history = model.chat(tokenizer=tokenizer, query=message, history=history,
                                       system="You are a helpful assistant.")
        history.append((message, response))
        print(f"MindChat: {response}")


def main(model_path: str):
    logging.debug(f"Into Main, model_path: '{model_path}'")
    # tokenizer, model = get_model(model_path)
    with gr.Blocks(title="🐋MindChat: 漫谈心理大模型", css="MindChat.css") as demo:
        with gr.Column():
            gr.Markdown(
                """
                🔎 MindChat(漫谈): 旨在通过营造轻松、开放的交谈环境, 以放松身心、交流感受或分享经验的方式, 为用户提供隐私、温暖、安全、及时、方便的对话环境, 从而帮助用户克服各种困难和挑战, 实现自我成长和发展.
                
                🦊 无论是在工作场所还是在个人生活中, MindChat期望通过自身的努力和专业知识, 在严格保护用户隐私的前提下, 全时段全天候为用户提供全面的心理陪伴和倾听, 同时实现自我成长和发展, 以期为建设一个更加健康、包容和平等的社会贡献力量.
                
                🙅‍ 目前，MindChat还不能替代专业的心理医生和心理咨询师，无法做出专业的心理诊断报告。虽MindChat在训练过程中极致注重模型安全和价值观正向引导，但仍无法保证模型输出正确且无害，内容上模型作者及平台不承担相关责任。
                
                👏 更为优质、安全、温暖的模型正在赶来的路上，欢迎关注：[MindChat Github](https://github.com/X-D-Lab/MindChat)
                """
            )
            gr.Chatbot()
            with gr.Row():
                translated_text = gr.Textbox(label="Target Text", lines=3, max_lines=10, interactive=True)
                submit_btn = gr.Button(value="发送", show_label=False, elem_classes="submit_btn")
            with gr.Row():
                retry_btn = gr.Button(value="🔄 重新生成", show_label=False, elem_classes="retry_btn")
                undo_btn = gr.Button(value="↩️ 撤销", show_label=False, elem_classes="undo_btn")
                clear_btn = gr.Button(value="🗑️ 清除历史", show_label=False, elem_classes="clear_btn")
    return demo


if __name__ == '__main__':
    log_set(log_level=logging.DEBUG, log_save=True)
    demo_ = main(model_path="/root/autodl-tmp/models/MindChat-Qwen-1_8B")

    demo_.queue(concurrency_count=2, max_size=15, status_update_rate="auto")
    demo_.launch(show_error=False, share=False, quiet=False, server_port=6006)
