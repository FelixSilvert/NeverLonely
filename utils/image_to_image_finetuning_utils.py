from diffusers import StableDiffusionXLPipeline
import torch


class StableDiffusionModel:
    def __init__(self, use_lora=False, lora_weights_path=None, base_model="stabilityai/stable-diffusion-xl-base-1.0"):
        """
        Initialize the Stable Diffusion model with optional LoRA weights.
        If no LoRA is provided, it uses the base model only.

        Args:
            use_lora (bool): Whether to load a LoRA model on top of the base model.
            lora_weights_path (str): Path to the LoRA weights file.
            base_model (str): Path to the base model, default is Stable Diffusion XL.
        """
        self.device = "cuda"
        self.base_model = base_model
        self.use_lora = use_lora
        self.lora_weights_path = lora_weights_path

        # Load the base model with memory optimizations
        self.text2img_pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16
        ).to(self.device)

        # Enable memory optimization
        self.text2img_pipe.enable_attention_slicing()  # Reduce memory usage
        self.text2img_pipe.enable_sequential_cpu_offload()  # Further reduce memory usage

        # Apply LoRA weights if specified
        if self.use_lora and self.lora_weights_path:
            self.load_lora_weights()

    def load_lora_weights(self):
        """Load LoRA weights onto the pipeline if specified."""
        print(f"Loading LoRA weights from {self.lora_weights_path}")
        self.text2img_pipe.load_lora_weights(self.lora_weights_path)

    def generate_text_to_image(self, prompt, negative_prompt=None, guidance_scale=7, num_inference_steps=20, seed=None):
        """
        Generate an image from a text prompt using the Text-to-Image pipeline.

        Args:
            prompt (str): The text prompt for image generation.
            negative_prompt (str): The negative prompt for avoiding certain attributes in the image.
            guidance_scale (float): The strength of the prompt influence. Default is 7.
            num_inference_steps (int): The number of diffusion steps. Default is 20 to save memory.
            seed (int): Optional seed for reproducibility. If None, random seed is used.
        """
        # Set the seed if specified for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            print(f"Using seed: {seed}")

        # Generate the image
        images = self.text2img_pipe(prompt, num_images_per_prompt=6, negative_prompt=negative_prompt,
                                    guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images
        return images
