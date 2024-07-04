import torch
import gc
from torch import nn
from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module
import bitsandbytes as bnb

def torch_gc():

    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
   
    gc.collect()

def restart_cpu_offload(pipe, load_mode):
    #if load_mode != '4bit' :
    #    pipe.disable_xformers_memory_efficient_attention()        
    optionally_disable_offloading(pipe)    
    gc.collect()
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()    
    #if load_mode != '4bit' :
    #    pipe.enable_xformers_memory_efficient_attention()  

def optionally_disable_offloading(_pipeline):

    """
    Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.

    Args:
        _pipeline (`DiffusionPipeline`):
            The pipeline to disable offloading for.

    Returns:
        tuple:
            A tuple indicating if `is_model_cpu_offload` or `is_sequential_cpu_offload` is True.
    """
    is_model_cpu_offload = False
    is_sequential_cpu_offload = False
    print(
            fr"Restarting CPU Offloading for {_pipeline.unet_name}..."
          )
    if _pipeline is not None:
        for _, component in _pipeline.components.items():
            if isinstance(component, nn.Module) and hasattr(component, "_hf_hook"):
                if not is_model_cpu_offload:
                    is_model_cpu_offload = isinstance(component._hf_hook, CpuOffload)
                if not is_sequential_cpu_offload:
                    is_sequential_cpu_offload = isinstance(component._hf_hook, AlignDevicesHook)

               
                remove_hook_from_module(component, recurse=True)

    return (is_model_cpu_offload, is_sequential_cpu_offload)

def quantize_4bit(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            device = child.weight.data.device

            # Create and configure the Linear layer
            has_bias = True if child.bias is not None else False
            
            # TODO: Make that configurable
            # fp16 for compute dtype leads to faster inference
            # and one should almost always use nf4 as a rule of thumb
            bnb_4bit_compute_dtype = torch.float16
            quant_type = "nf4"

            new_layer = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=has_bias,
                compute_dtype=bnb_4bit_compute_dtype,
                quant_type=quant_type,
            )

            new_layer.load_state_dict(child.state_dict())
            new_layer = new_layer.to(device)

            # Set the attribute
            setattr(module, name, new_layer)
        else:
            # Recursively apply to child modules
            quantize_4bit(child)