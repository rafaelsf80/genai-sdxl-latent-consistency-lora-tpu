# LCM Scheduler with Flax and TPUs

***** WORK IN PROGRES *****

***** SEE OPEN ISSUE BELOW *****

Following this [blog post](https://huggingface.co/blog/lcm_lora), this repo shows how to use the Flax adaptation of the LCM Scheduler to use LoRA and any SDXL diffusion model. 

The codesamples are tested with TPU v5e.


## Usage of non-Flax LCM Scheduler

Refer to the previous [blog post](https://huggingface.co/blog/lcm_lora). GPU only.


## Usage of Flax LCM Scheduler 

Refer to this [code example](./lora_sd_tpu_inference.py). TPU only. Note the open issue below.

Requirements:

1. Install TPU v5e requirements:
```sh
sudo apt update
sudo apt install python3.8-venv
python -m venv tpuenv
source tpuenv/bin/activate
pip install -r requirements_tpu.txt
```

2. Clone the updated diffusers library with the flax implementation of LCMScheduler (FlaxLCMScheduler class). 
```py
git clone https://github.com/rafaelsf80/diffusers/tree/f/flax_lora_lcm_scheduler
```

3. Move the new diffusers code to the proper diretory in order to overwrite the previous diffusers installation (Example: `./tpuenv/lib/python3.10/site-packages/diffusers/``)

4. Usage:
```sh
python lora_sd_tpu_inference.py
```


## Open issue

At inference time, the following traceback is generated:
```sh
Traceback (most recent call last):
  File "/home/rafaelsanchez_jcatalu_altostrat_/sd_benchmark/lora_sd_tpu_one_run.py", line 107, in <module>
    generate(default_prompt, default_neg_prompt)
  File "/home/rafaelsanchez_jcatalu_altostrat_/sd_benchmark/lora_sd_tpu_one_run.py", line 91, in generate
    images = pipeline(
  File "/home/rafaelsanchez_jcatalu_altostrat_/sd_benchmark/tpuenv/lib/python3.10/site-packages/diffusers/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py", line 116, in __call__
    images = _p_generate(
  File "/home/rafaelsanchez_jcatalu_altostrat_/sd_benchmark/tpuenv/lib/python3.10/site-packages/diffusers/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py", line 298, in _p_generate
    return pipe._generate(
  File "/home/rafaelsanchez_jcatalu_altostrat_/sd_benchmark/tpuenv/lib/python3.10/site-packages/diffusers/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py", line 265, in _generate
    latents, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (latents, scheduler_state))
  File "/home/rafaelsanchez_jcatalu_altostrat_/sd_benchmark/tpuenv/lib/python3.10/site-packages/diffusers/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py", line 257, in loop_body
    latents, scheduler_state = self.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
  File "/home/rafaelsanchez_jcatalu_altostrat_/sd_benchmark/tpuenv/lib/python3.10/site-packages/diffusers/schedulers/scheduling_lcm_flax.py", line 631, in step
    self._init_step_index(timestep)
  File "/home/rafaelsanchez_jcatalu_altostrat_/sd_benchmark/tpuenv/lib/python3.10/site-packages/diffusers/schedulers/scheduling_lcm_flax.py", line 308, in _init_step_index
    index_candidates = (self.timesteps == timestep).nonzero()
  File "/home/rafaelsanchez_jcatalu_altostrat_/sd_benchmark/tpuenv/lib/python3.10/site-packages/diffusers/configuration_utils.py", line 138, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'FlaxLCMScheduler' object has no attribute 'timesteps'. Did you mean: 'set_timesteps'?
```


## References

1. [Sample inference](./sd_tpu_inference.py) for SDXL in TPU v5e. No LoRA. TPU only.

2. The `.save_pretrained()` method on the Scheduler saves the following output `scheduler_config.json`:
```json
{
  "_class_name": "LCMScheduler",
  "_diffusers_version": "0.24.0",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": false,
  "clip_sample_range": 1.0,
  "dynamic_thresholding_ratio": 0.995,
  "interpolation_type": "linear",
  "num_train_timesteps": 1000,
  "original_inference_steps": 50,
  "prediction_type": "epsilon",
  "rescale_betas_zero_snr": false,
  "sample_max_value": 1.0,
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "thresholding": false,
  "timestep_scaling": 10.0,
  "timestep_spacing": "leading",
  "trained_betas": null,
  "use_karras_sigmas": false
}
```