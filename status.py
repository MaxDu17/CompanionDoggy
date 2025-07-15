
# import gradio as gr

# with gr.Blocks() as demo:
#     Image = gr.Image("green.jpg")
#     Image = gr.Image("yellow.jpg")
#     Image = gr.Image("red.jpg")
# demo.launch()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# import gradio as gr
# from PIL import Image
# import numpy as np
    
# def status(tracking):
#     if int(tracking) > 50:
#         return "Yes"
#     else:
#         return "No"

# with gr.Blocks() as demo:
#     tracking = gr.Slider(label = "tracking")
#     greet_btn = gr.Button("Test")
#     output = gr.Textbox(label="Output Box")
#     greet_btn.click(fn=status, inputs=tracking, outputs=output, api_name="Test")

#     demo.launch()


import gradio as gr
from PIL import Image
import numpy as np

tracking = 77
# random values for tracking: 
    # 0 - 30: Not tracking
    # 30 - 60: Inconsistent tracking
    # 60+ : Tracking effectively

def status():
    if tracking > 60:
        return "green.jpg"
    elif tracking > 30:
        return "yellow.jpg"
    else:
        return "red.jpg"

with gr.Blocks() as demo:
    status_output = gr.Image("pending.png")
    status_btn = gr.Button("Test Status")
    status_btn.click(fn=status, outputs=status_output)

    demo.launch()



