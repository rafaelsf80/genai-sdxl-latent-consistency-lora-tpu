# Sample inference for SDXL in TPU v5e

import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from diffusers import FlaxStableDiffusionXLPipeline
import time

from datasets import load_dataset
import random 

# Let's cache the model compilation, so that it doesn't take as long the next time around.
from jax.experimental.compilation_cache import compilation_cache as cc

cc.initialize_cache("/tmp/sdxl_cache")

num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"Found {num_devices} JAX devices of type {device_type}.")
assert "TPU" in device_type, "Available device is not a TPU, please select TPU from Runtime > Change runtime type > Hardware accelerator"


pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", split_head_dim=True
)

scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params["scheduler"] = scheduler_state

default_prompt = "high-quality photo of a baby dolphin ​​playing in a pool and wearing a party hat"
default_neg_prompt = "illustration, low-quality"
default_seed = 33
default_guidance_scale = 5.0
default_num_steps = 30

def tokenize_prompt(prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    return prompt_ids, neg_prompt_ids

NUM_DEVICES = jax.device_count()

# Model parameters don't change during inference,
# so we only need to replicate them once.
p_params = replicate(params)

def replicate_all(prompt_ids, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_neg_prompt_ids, rng

def generate(
    prompt,
    negative_prompt,
    seed=default_seed,
    guidance_scale=default_guidance_scale,
    num_inference_steps=default_num_steps,
):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    images = pipeline(
        prompt_ids,
        p_params,
        rng,
        num_inference_steps=num_inference_steps,
        neg_prompt_ids=neg_prompt_ids,
        guidance_scale=guidance_scale,
        jit=True,
    ).images

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1], ) + images.shape[-3:])
    return pipeline.numpy_to_pil(np.array(images))

start = time.time()
print(f"Compiling ...")
generate(default_prompt, default_neg_prompt)
print(f"Compiled in {time.time() - start}")

dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="test")

dts = []
i = 0
for x in range(2):
    random_index = random.randint(0, len(dataset) - 1)
    
    start = time.time()
    prompt = dataset[random_index]["Prompt"]
    neg_prompt = "cartoon, illustration, animation"

    print(f"Prompt: {prompt}")
    images = generate(prompt, neg_prompt)
    t = time.time() - start
    print(f"Inference in {t}")

    dts.append(t)
    for img in images:
        img.save(f'{i:06d}.jpg')
        i += 1    

mean = np.mean(dts)
stdev = np.std(dts)
print(f"batches: {i},  Mean {mean:.2f} sec/batch± {stdev * 1.96 / np.sqrt(len(dts)):.2f} (95%)")



