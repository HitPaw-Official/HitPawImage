import os
import time, datetime
import gradio as gr
import cv2
import numpy as np
from zipfile import ZipFile

from human_parsing import HumanParsing

hp = HumanParsing()

output_file = '/home/ubuntu02/liuji/projects/real-ESRGAN/outputs/gradio_demo/results.zip'

def main(files):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join('outputs/gradio_demo', now_str)
    os.makedirs(save_dir, exist_ok=True)
    txt_time = open(os.path.join(save_dir, "000_running_time.txt"), 'w')
    for idx, file in enumerate(files):
        t0 = time.time()
        input_path = file.name
        img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), 1)
        if img is None:
            continue
        name = os.path.basename(input_path)
        output_path = os.path.join(save_dir, name)
        out = hp.run_with_detect(img)
        cv2.imwrite(output_path, out)
        info = "reference image: {}, using time: {:.2f} s \n".format(name, time.time()-t0)
        txt_time.write(info)
            
    txt_time.close()
    with ZipFile(output_file, "w") as zipObj:
        for name in os.listdir(save_dir):
            file_path = os.path.join(save_dir, name)
            zipObj.write(file_path, name)
    return output_file


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("""
            Human Parsing Demo
        """)
    with gr.Row():
        gr.Markdown('''
                    - Step 1. 上传图像, 可批量导入
                    - Step 2. 点击运行，等待算法输出
                        ''')
    with gr.Row():
        input_image = gr.File(file_count="multiple")
    with gr.Row():    
        gr_btn = gr.Button(value='运行')
    with gr.Row():
        output_image = gr.File()

    gr_btn.click(
    fn=main,
    inputs=[
            input_image,
    ],
    outputs=[output_image],
    )

demo.launch(server_name='0.0.0.0', server_port=7905, show_error=True)    