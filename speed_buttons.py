import gradio as gr

speed = 3

def speedinc():
    global speed
    speed += 1
    return "Current Speed: " + str(speed) + "m/s"

def speeddec():
    global speed
    speed -= 1
    return "Current Speed: " + str(speed) + "m/s"

#def current_speed()
    #return "Current Speed: " + str(speed) + "m/s"

output = gr.Textbox("Speed")

with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column(scale=2):
            speedinc_btn = gr.Button("Speed Up", scale=5)
            speedinc_btn.click(fn=speedinc, outputs=output)

            speeddec_btn = gr.Button("Speed Down", scale=5)
            speeddec_btn.click(fn=speeddec, outputs=output)

        with gr.Column():
            output.render()

    demo.launch()