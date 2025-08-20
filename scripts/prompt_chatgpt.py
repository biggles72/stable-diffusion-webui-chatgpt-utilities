import copy
import os

import modules.scripts as scripts
import gradio as gr
from modules import images, shared, script_callbacks
from modules.processing import Processed, process_images
from modules.shared import state
import modules.sd_samplers
from scripts.chatgpt_answers import get_ollama_answers
from scripts.template_utils import get_templates
from scripts.ollama_list_models import list_ollama_models

script_dir = scripts.basedir()

class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.ollama_model = "llama2"
        
    def title(self):
        return "Ollama"

    def ui(self, is_img2img):
        templates = get_templates(os.path.join(script_dir, "templates"))

        with gr.Row():
            templates_dropdown = gr.Dropdown(
                label="Templates", 
                choices=[t[0] for t in templates],
                type="index", 
                elem_id="ollama_template_dropdown")
            # Ollama model selection (dynamic)
            ollama_models = list_ollama_models()
            ollama_model_dropdown = gr.Dropdown(
                label="Ollama Model",
                choices=ollama_models if ollama_models else ["llama2"],
                value=ollama_models[0] if ollama_models else "llama2",
                type="value",
                elem_id="ollama_model_dropdown")
            precision_dropdown = gr.Dropdown(
                label="Answer precision",
                choices=["Specific", "Normal", "Dreamy", "Hallucinating"],
                value="Dreamy",
                type="index",
            )

        ollama_prompt = gr.Textbox(label="", placeholder="Ollama prompt (Try some templates for inspiration)", lines=4)

        with gr.Row():
            ollama_batch_count = gr.Number(value=4, label="Response count")
            ollama_append_to_prompt = gr.Checkbox(label="Append to original prompt instead of replacing it", default=False)

        with gr.Row():
            ollama_prepend_prompt = gr.Textbox(label="Prepend generated prompt with", lines=1)
            ollama_append_prompt = gr.Textbox(label="Append generated prompt with", lines=1)

        with gr.Row():
            ollama_no_iterate_seed = gr.Checkbox(label="Don't increment seed per permutation", default=False)
            ollama_generate_original_prompt = gr.Checkbox(label="Generate original prompt also", default=False)

        with gr.Row():
            ollama_generate_debug_prompt = gr.Checkbox(label="DEBUG - Stop before image generation", default=False)
            ollama_just_run_prompts = gr.Checkbox(label="OVERRIDE - Run prompts from textbox (one per line)", default=False)

        with gr.Row():
            gr.HTML('<a href="https://github.com/hallatore/stable-diffusion-webui-chatgpt-utilities" target="_blank" style="text-decoration: underline;">Help & More Examples</a>')

        def apply_template(dropdown_value, prompt, append_to_prompt):
            if not (isinstance(dropdown_value, int)):
                return prompt, append_to_prompt

            file_path = templates[dropdown_value][1]
            dir_name = os.path.basename(os.path.dirname(file_path))
            
            with open(templates[dropdown_value][1], 'r') as file:
                template_text = file.read()

            return template_text, dir_name.lower() == "append"

        templates_dropdown.change(
            apply_template, 
            inputs=[templates_dropdown, ollama_prompt, ollama_append_to_prompt], 
            outputs=[ollama_prompt, ollama_append_to_prompt]
        )
        
        return [
            ollama_prompt, 
            precision_dropdown,
            ollama_batch_count, 
            ollama_append_to_prompt, 
            ollama_prepend_prompt, 
            ollama_append_prompt, 
            ollama_no_iterate_seed, 
            ollama_generate_original_prompt,
            ollama_generate_debug_prompt,
            ollama_just_run_prompts,
            ollama_model_dropdown
        ]

    def run(
            self, 
            p, 
            ollama_prompt, 
            precision_dropdown,
            ollama_batch_count, 
            ollama_append_to_prompt, 
            ollama_prepend_prompt, 
            ollama_append_prompt,
            ollama_no_iterate_seed, 
            ollama_generate_original_prompt,
            ollama_generate_debug_prompt,
            ollama_just_run_prompts,
            ollama_model_dropdown
        ):
        modules.processing.fix_seed(p)

        model = ollama_model_dropdown or "llama2"

        if (ollama_prompt == ""):
            raise Exception("Ollama prompt is empty.")

        if (ollama_batch_count < 1):
            raise Exception("Ollama batch count needs to be 1 or higher.")

        temperature = 1.0
        if (precision_dropdown == 0):
            temperature = 0.5
        elif (precision_dropdown == 1):
            temperature = 1.0
        elif (precision_dropdown == 2):
            temperature = 1.25
        elif (precision_dropdown == 3):
            temperature = 1.5

        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt
        prompts = []

        if (ollama_just_run_prompts):
            for prompt in ollama_prompt.splitlines():
                prompts.append([prompt, prompt])
        else:
            ollama_answers = get_ollama_answers(
                ollama_prompt, int(ollama_batch_count), temperature, original_prompt, model=model
            )
            ollama_prefix = ""

            if len(original_prompt) > 0:
                if ollama_generate_original_prompt:
                    prompts.append(["", original_prompt])

                if ollama_append_to_prompt:
                    ollama_prefix = f"{original_prompt}, "

            for answer in ollama_answers:
                prompts.append([answer, f"{ollama_prefix}{ollama_prepend_prompt}{answer}{ollama_append_prompt}"])

            print(f"Prompts:\r\n" + "\r\n".join([p[1] for p in prompts]) + "\r\n")

            if (ollama_generate_debug_prompt):
                raise Exception("DEBUG - Stopped before image generation.\r\n\r\n" + "\r\n".join([p[1] for p in prompts]))

        p.do_not_save_grid = True
        state.job_count = 0
        permutations = 0
        state.job_count += len(prompts) * p.n_iter
        permutations += len(prompts)
        print(f"Creating {permutations} image permutations")
        image_results = []
        all_prompts = []
        infotexts = []
        current_seed = p.seed

        for prompt in prompts:
            copy_p = copy.copy(p)
            copy_p.prompt = prompt[1]
            copy_p.seed = current_seed
            if not ollama_no_iterate_seed:
                current_seed += 1
            proc = process_images(copy_p)
            temp_grid = images.image_grid(proc.images, p.batch_size)
            image_results.append(temp_grid)
            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        if (len(prompts) > 1):
            grid = images.image_grid(image_results, p.batch_size)
            infotexts.insert(0, infotexts[0])
            image_results.insert(0, grid)
            images.save_image(grid, p.outpath_grids, "grid", grid=True, p=p)

        return Processed(p, image_results, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)