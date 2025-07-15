
import gradio as gr
import numpy as np

speed = 3
tracking = 80
# random values for tracking: 
    # 0 - 30: Not tracking
    # 30 - 60: Inconsistent tracking
    # 60+ : Tracking effectively

css = """
.speed-text h1 {
    text-align: center; 
    font-size: 80px !important;
}
.speed-subtext {
    text-align: center; 
}
"""


def status():
    if tracking > 60:
        return "GUIassets/green.jpg"
    elif tracking > 30:
        return "GUIassets/yellow.jpg"
    else:
        return "GUIassets/red.jpg"

def speedinc():
    global speed
    speed += 1
    return "# " + str(speed) + "m/s"

def speeddec():
    global speed
    speed -= 1
    return "# " + str(speed) + "m/s"

output = gr.Markdown("# Speed", elem_classes="speed-text")

with gr.Blocks(css=css) as demo:
    with gr.Row():      
        with gr.Column(scale=3):
            status_output = gr.Image("GUIassets/pending.png")
            status_btn = gr.Button("Test Status")
            status_btn.click(fn=status, outputs=status_output)
         
        with gr.Column(min_width=400):
            speedinc_btn = gr.Button("Speed Up", scale=1)
            speedinc_btn.click(fn=speedinc, outputs=output)
            speed_title = gr.Markdown("## Current speed:", elem_classes="speed-subtext")
            output.render()
            speeddec_btn = gr.Button("Speed Down", scale=1)
            speeddec_btn.click(fn=speeddec, outputs=output)
   

demo.launch()

