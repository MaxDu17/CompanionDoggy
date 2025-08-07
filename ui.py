
import gradio as gr
import numpy as np
from global_state import GlobalState

class GUI:
    def __init__(self, global_state: GlobalState):
        self.speed = 3
        self.person_distance = 80
        self.global_state = global_state
        self.css = """
        .speed-text h1 {
            text-align: center; 
            font-size: 80px !important;
        }
        """

        self.build_gui()

    def status(self):
        if self.person_distance > 60:
            return "GUIassets/green.jpg"
        elif self.person_distance > 30:
            return "GUIassets/yellow.jpg"
        else:
            return "GUIassets/red.jpg"

    def speedinc(self):
        self.speed += 0.25
        self.global_state.lock_set("speed", self.speed)

        return "# " + str(self.speed) + "min/mile"
    
    def speeddec(self):
        self.speed -= 0.25
        self.global_state.lock_set("speed", self.speed)
        return "# " + str(self.speed) + "min/mile"

    def build_gui(self):
        output = gr.Markdown("# Speed", elem_classes="speed-text")

        with gr.Blocks(css=self.css) as self.demo:
            with gr.Row():      
                with gr.Column(scale=3):
                    status_output = gr.Image("GUIassets/pending.png")
                    status_btn = gr.Button("Test Status")
                    status_btn.click(fn=self.status, outputs=status_output)
                
                with gr.Column(min_width=400):
                    speedinc_btn = gr.Button("Speed Up", scale=1)
                    speedinc_btn.click(fn=self.speedinc, outputs=output)
                    speed_title = gr.Markdown("## Current speed:", elem_classes="speed-subtext")
                    output.render()
                    speeddec_btn = gr.Button("Speed Down", scale=1)
                    speeddec_btn.click(fn=self.speeddec, outputs=output)

    def gui(self):
        print("Starting Gradio UI")
        self.demo.queue().launch()

if __name__ == "__main__":
    global_state = GlobalState()
    gui = GUI(global_state)
    gui.gui()