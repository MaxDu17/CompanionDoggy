
import gradio as gr
import numpy as np
from global_state import GlobalState

class GUI:
    def __init__(self, global_state: GlobalState):
        self.speed = 3
        self.person_distance = 80
        self.global_state = global_state
        self.update_freq = 0.2
        self.css = """
        .speed-text h1 {
            text-align: center; 
            font-size: 80px !important;
        }
        """

        self.img_size = (300, 700)
        self.green_img = np.zeros(self.img_size + (3,), dtype=np.uint8)
        self.green_img[:, :, 0] = 0
        self.green_img[:, :, 1] = 255
        self.green_img[:, :, 2] = 0
        self.red_img = np.zeros(self.img_size + (3,), dtype=np.uint8)
        self.red_img[:, :, 0] = 255
        self.red_img[:, :, 1] = 0
        self.red_img[:, :, 2] = 0
        self.yellow_img = np.zeros(self.img_size + (3,), dtype=np.uint8)
        self.yellow_img[:, :, 0] = 255
        self.yellow_img[:, :, 1] = 255
        self.yellow_img[:, :, 2] = 0

        self.build_gui()

    def status(self):
        if self.person_distance is None:
            return self.red_img
        elif self.person_distance < 1200:
            return self.green_img
        elif self.person_distance >= 1200:
            return self.yellow_img

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
                    vision_img = gr.Image(
                                self.status,
                                type="numpy",
                                show_label=False,
                                show_download_button=False,
                                interactive=False,
                                min_width=200,
                                every=self.update_freq,
                            )

                    
                    # Timer to auto-check every 2 seconds (adjust as needed)
                    # timer = gr.Timer()
                    # timer.run(fn=self.status, outputs=self.status_output, interval=0.5, every = True)
                    timer = gr.Timer() # interval=0.5, repeat=True)
                    timer.tick(fn=self.status, inputs=None) #, outputs=status_output)
                        
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