import gradio as gr
from PIL import Image
import numpy as np
from numpy import random
import time

tracking = 77
# random values for tracking: 
    # 0 - 30: Not tracking
    # 30 - 60: Inconsistent tracking
    # 60+ : Tracking effectively

def status():
    global tracking
    #time.sleep(0.2) #optional delay
    if tracking > 60:
        return "GUIassets/green.jpg"
    elif tracking > 30:
        return "GUIassets/yellow.jpg"
    else:
        return "GUIassets/red.jpg"

number = random.randint(100)
print("bello")


# def update():
#     global tracking
#     tracking = random.randint(100)
#     print(str(tracking))
#     time.sleep(0.5)
#     return status()


# with gr.Blocks() as demo:
#     with gr.Row():
#         status_output = gr.Image("pending.png")
#         with gr.Column():
#             number = gr.Markdown(str(tracking))
#             run_random = gr.Button("run random")
#             run_random.click(fn=update, outputs=status_output)
#     demo.launch()

# while True:
#     update()
#     with gr.Blocks() as demo:
#         with gr.Row():
#             status_output = gr.Image("pending.png")
#             number = gr.Markdown(str(tracking))
#     print("updating...")
#     time.sleep(0.5)
#     demo.launch()


def update():
    global tracking
    tracking = random.randint(100)
    print(str(tracking)) #Terminal feedback
    time.sleep(0.5)
    return status(), "# " + str(tracking)

def test():
    number = random.randint(100)
    print(number)

with gr.Blocks() as demo:
    with gr.Row():
        status_output = gr.Image(status)
        with gr.Column():
            wurd = gr.Markdown("### Tracking Status:")
            number = gr.Markdown("# " + str(tracking))
            loop = gr.Timer(value=1)
            loop.tick(fn=update, outputs=[status_output, number])
    demo.launch()