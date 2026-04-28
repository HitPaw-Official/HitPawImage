from safetensors.torch import load_file, save_file
from lora import create_network_from_weights
# diffusers==0.14.0
from diffusers import StableDiffusionPipeline
import torch


def apply_lora(pipe, lora_path, output_path, weight:float = 1.0):
    
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    sd = load_file(lora_path)
    lora_network, sd = create_network_from_weights(weight, None, vae, text_encoder, unet, sd)
    lora_network.apply_to(text_encoder, unet)
    lora_network.load_state_dict(sd)
    lora_network.to("cpu", dtype=torch.float16)

    pipe.save_pretrained(output_path)
    # save_file(lora_network.state_dict(), output_path)


if __name__ == "__main__":
    pipe = StableDiffusionPipeline.from_pretrained("checkpoint/v2.0/artyou", torch_dtype=torch.float16)
    apply_lora(pipe, "checkpoint/v2.0/barbiev1.safetensors", "checkpoint/v2.0/18_barbiev1", 0.8)
