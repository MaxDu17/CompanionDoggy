
import gradio as gr
import numpy as np
from global_state import GlobalState

class GUI:
    def __init__(self, global_state: GlobalState):
        self.global_state = global_state
        self.update_freq = 0.2
        self.css = """
        .speed-text h2 {
            text-align: center; 
            font-size: 40px !important;
        }
        .mode-text h2 {
            text-align: center; 
        }
        """

        self.img_size = (700, 700)
        self.green_img = np.zeros(self.img_size + (3,), dtype=np.uint8)
        self.green_img[:, :, 1] = 255
        self.red_img = np.zeros(self.img_size + (3,), dtype=np.uint8)
        self.red_img[:, :, 0] = 255
        self.yellow_img = np.zeros(self.img_size + (3,), dtype=np.uint8)
        self.yellow_img[:, :, 0] = 255
        self.yellow_img[:, :, 1] = 255

        self.build_gui()

    def status(self):
        person_distance = self.global_state.lock_get("person_distance")
        if person_distance is None:
            return self.red_img
        elif person_distance < 1200:
            return self.green_img
        elif person_distance >= 1200:
            return self.yellow_img

    def speedinc(self):
        speed = self.global_state.lock_get("speed")
        speed += 0.5 
        self.global_state.lock_set("speed", speed)

        return "## " + str(speed) + "m/s"
    
    def speeddec(self):
        speed = self.global_state.lock_get("speed")
        speed -= 0.5 
        self.global_state.lock_set("speed", speed)
        return "## " + str(speed) + "m/s"

    def build_gui(self):
        # speed_value_output = gr.Markdown(f"## {self.global_state.lock_get('speed')}", elem_classes="speed-text", every = self.update_freq)

        # status_output = gr.Markdown(f"## OPERATION MODE: {self.global_state.lock_get('mode')}", elem_classes="mode-text", every = self.update_freq)

        # output = gr.Markdown("# Speed", elem_classes="speed-text")

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

                
                                        
                with gr.Column(min_width=400):
                    status_output = gr.Markdown(lambda: f"## OPERATION MODE: {self.global_state.lock_get('mode')}", elem_classes="mode-text", every = 1)
                    speedinc_btn = gr.Button("Speed Up", scale=1)
                    speedinc_btn.click(fn=self.speedinc) # , outputs=speed_value_output)
                    speed_title = gr.Markdown(lambda: "## Current speed:", elem_classes="speed-subtext")
                    speed_value_output = gr.Markdown(lambda: f"## {self.global_state.lock_get('speed')}", elem_classes="speed-text", every = 1)

                    speeddec_btn = gr.Button("Slow Down", scale=1)
                    speeddec_btn.click(fn=self.speeddec) #, outputs=speed_value_output)

    def gui(self):
        print("Starting Gradio UI")
        self.demo.queue().launch()

if __name__ == "__main__":
    global_state = GlobalState()
    gui = GUI(global_state)
    gui.gui()