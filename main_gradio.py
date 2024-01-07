import argparse
import logging
import os
import time
from typing import List

import gradio as gr
from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from tools.logging_utils import log_set


def load_model(model_path: str = "X-D-Lab/MindChat-Qwen-7B"):
    logging.debug(f"Starting Loading {os.path.basename(model_path)}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True,
                                                 fp16=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    logging.info(f"Loading {os.path.basename(model_path)} finished")
    return tokenizer, model


class MindChat(object):
    def __init__(self, model_path: str = None, cache_dir: str = "./", model_id: str = "X-D-Lab/MindChat-Qwen-1_8B"):
        self.model_path = snapshot_download(model_id=model_id, cache_dir=cache_dir,
                                            revision='v1.0.0') if model_path is None else model_path
        self.tokenizer, self.model = load_model(model_path)

    def get_response(self, message: str, history: List[List[str]]) -> str:
        logging.debug(f"Start a chat, message: '{message}', history: '{history}'")
        history = history[-20:]  # todo: 暂时采用[-20:]防止token超限, 待更换为tokenizer
        response, _ = self.model.chat(
            tokenizer=self.tokenizer,
            query=message,
            history=[] if history is None else history,
        )
        logging.debug(f"Finish a chat, response: '{response}'")
        return response

    def chat(self, message: str = "what can you do？", history: List[List[str]] = None):
        history = [] if history is None else history
        response = self.get_response(message=message, history=history)
        history.append([message, ""])
        for character in response:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

    def retry(self, history: List[List[str]]):
        """从history列表中重新执行最后一次对话"""
        logging.debug(f"Retry history: {history}")
        if history:
            message = history[-1][0]
            history = history[:-1]
            response = self.get_response(message=message, history=history)
            history.append([message, ""])
            for character in response:
                history[-1][1] += character
                time.sleep(0.05)
                yield history
        else:
            return []

    @staticmethod
    def undo(history: List[List[str]]) -> List[List[str]]:
        """从history列表中删除最后一次对话"""
        return history[:-1]


def main(mind_chat: MindChat):
    with gr.Blocks(title="🐋MindChat: 漫谈心理大模型") as demo:
        with gr.Column():
            gr.Markdown(
                """
                # 🐋MindChat: 漫谈心理大模型
                
                🔎 MindChat(漫谈): 旨在通过营造轻松、开放的交谈环境, 以放松身心、交流感受或分享经验的方式, 为用户提供隐私、温暖、安全、及时、方便的对话环境, 从而帮助用户克服各种困难和挑战, 实现自我成长和发展。
            
                🦊 无论是在工作场所还是在个人生活中, MindChat期望通过自身的努力和专业知识, 在严格保护用户隐私的前提下, 全时段全天候为用户提供全面的心理陪伴和倾听, 同时实现自我成长和发展, 以期为建设一个更加健康、包容和平等的社会贡献力量。
                
                🙅‍ 目前，MindChat还不能替代专业的心理医生和心理咨询师，无法做出专业的心理诊断报告。虽MindChat在训练过程中极致注重模型安全和价值观正向引导，但仍无法保证模型输出正确且无害，内容上模型作者及平台不承担相关责任。
                
                👏 更为优质、安全、温暖的模型正在赶来的路上，欢迎关注：[MindChat Github](https://github.com/X-D-Lab/MindChat)
                """
            )
            chatbot = gr.Chatbot([], elem_id="chatBot", bubble_full_width=False)
            with gr.Row():
                message_text = gr.Textbox(placeholder="Type a message...", lines=1, max_lines=10, interactive=True,
                                          scale=6, show_label=False)
                submit_btn = gr.Button(value="发送", show_label=False, elem_classes="submitBtn", scale=2)
            with gr.Row():
                retry_btn = gr.Button(value="🔄 重新生成", show_label=False, elem_classes="retryBtn")
                undo_btn = gr.Button(value="↩️ 撤销", show_label=False, elem_classes="undoBtn")
                clear_btn = gr.Button(value="🗑️ 清除历史", show_label=False, elem_classes="clearBtn")

        message_text.submit(mind_chat.chat, inputs=[message_text, chatbot], outputs=[chatbot])
        submit_btn.click(mind_chat.chat, inputs=[message_text, chatbot], outputs=[chatbot])
        retry_btn.click(mind_chat.retry, inputs=[chatbot], outputs=[chatbot])
        undo_btn.click(mind_chat.undo, inputs=[chatbot], outputs=[chatbot])
        clear_btn.click(lambda *args: [], inputs=[chatbot], outputs=[chatbot])
    return demo


if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser("MindChat Gradio WebUI")
    parser.add_argument("--model_path", "-m", type=str, help="model path", nargs='?', default=None)
    parser.add_argument("--model_id", "-mid", type=str, help="modelscope model id", nargs='?',
                        default="X-D-Lab/MindChat-Qwen-1_8B")
    parser.add_argument("--cache_dir", "-c", type=str, help="cache dir", nargs='?', default="./")
    parser.add_argument("--port", "-p", type=int, help="server port", nargs='?', default=6006)
    args_ = parser.parse_args()

    # logging
    log_set(log_level=logging.INFO, log_save=True)

    # Gradio Page
    demo_ = main(mind_chat=MindChat(model_path=args_.model_path, model_id=args_.model_id, cache_dir=args_.cache_dir))

    demo_.queue(concurrency_count=2, max_size=15, status_update_rate="auto")
    demo_.launch(show_error=False, share=False, quiet=True, server_port=args_.port, debug=False, show_api=False)
