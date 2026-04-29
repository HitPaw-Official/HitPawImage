import gradio as gr
import cv2
from human_parsing import HumanParsing

hp = HuamnParsing()

def main(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = hp.run_with_detect(img)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("""
            Human Parsing Demo
        """)
    with gr.Row():
        gr.Markdown('''
                    - Step 1. 上传图像
                    - Step 2. 点击运行，等待算法输出
                        ''')
    with gr.Row():
        input_image = gr.Image(source='upload', type="numpy")
    with gr.Row():    
        gr_btn = gr.Button(value='运行')
    with gr.Row():
        output_image = gr.Image()

    gr_btn.click(
    fn=main,
    inputs=[
            input_image,
    ],
    outputs=[output_image],
    )

demo.launch(server_name='0.0.0.0', server_port=7905, show_error=True)    