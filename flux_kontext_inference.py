"""
Flux Kontext with FluxSpace - Simple Inference

Usage:
    python flux_kontext_inference.py

This directly injects FluxSpace blocks into Flux Kontext transformer.
"""

import torch
import numpy as np
from PIL import Image
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

# Import FluxSpace blocks
import sys
sys.path.insert(0, "src/diffusers/pipelines/flux")
from flux_kontext_blocks import register_fluxspace_blocks


class FluxKontextFluxSpace:
    """
    Flux Kontext with FluxSpace Enhancement
    
    Directly injects FluxSpace transformer blocks for:
    - Orthogonal Edit Direction Injection
    - Self-Supervised Spatial Masking
    """
    
    def __init__(self, model_id="black-forest-labs/FLUX.1-Kontext-dev", 
                 device="cuda", dtype=torch.bfloat16):
        print(f"Loading {model_id}...")
        self.pipe = FluxKontextPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to(device)
        self.device = device
        self.dtype = dtype
        
        # Inject FluxSpace blocks into transformer
        print("Injecting FluxSpace blocks...")
        n_double, n_single = register_fluxspace_blocks(self.pipe.transformer)
        print(f"Registered {n_double} double blocks, {n_single} single blocks")
        
    def encode_prompt(self, prompt):
        """Encode a prompt to get T5 and CLIP embeddings"""
        t5_embeds = self.pipe._get_t5_prompt_embeds(prompt=prompt, device=self.device)
        clip_embeds = self.pipe._get_clip_prompt_embeds(prompt=prompt, device=self.device)
        return t5_embeds, clip_embeds
    
    def get_temb(self, timestep, pooled_embeds, guidance):
        """Get time embedding for projections"""
        if self.pipe.transformer.config.guidance_embeds:
            return self.pipe.transformer.time_text_embed(timestep, guidance, pooled_embeds)
        return self.pipe.transformer.time_text_embed(timestep, pooled_embeds)
    
    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        edit_prompt: str = None,
        edit_scale: float = 3.0,
        edit_global_scale: float = 0.0,
        attention_threshold: float = 0.5,
        edit_start_step: int = 0,
        edit_stop_step: int = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 2.5,
        seed: int = None,
        **kwargs
    ):
        """
        Generate edited image using FluxSpace techniques.
        
        Args:
            image: Input image
            prompt: Base prompt describing the image
            edit_prompt: Edit instruction (e.g., "eyeglasses", "smile")
            edit_scale: Fine-grained edit strength (λ_fine)
            edit_global_scale: Coarse/global edit strength (λ_coarse) [0-1]
            attention_threshold: Spatial mask threshold (τ_m)
            edit_start_step: When to start applying edit
            edit_stop_step: When to stop applying edit
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            seed: Random seed
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # If no edit_prompt, use standard pipeline
        if not edit_prompt:
            return self.pipe(
                image=image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                **kwargs
            ).images[0]
        
        # Encode prompts
        edit_t5, edit_clip = self.encode_prompt(edit_prompt)
        null_t5, null_clip = self.encode_prompt("")
        base_t5, base_clip = self.encode_prompt(prompt)
        
        # Coarse editing: Orthogonal projection on CLIP pooled embeddings
        if edit_global_scale > 0:
            emb_product = torch.sum(edit_clip * base_clip, dim=1, keepdim=True)
            emb_norm_sq = torch.sum(base_clip * base_clip, dim=1, keepdim=True)
            emb_projection = (emb_product / (emb_norm_sq + 1e-10)) * base_clip
            ortho_emb = edit_clip - emb_projection
            edited_pooled = (1 - edit_global_scale) * base_clip + edit_global_scale * ortho_emb
        else:
            edited_pooled = base_clip
        
        # Setup FluxSpace parameters in joint_attention_kwargs
        if edit_stop_step is None:
            edit_stop_step = num_inference_steps
        
        # Pre-compute context embeddings
        edit_context_embeds = self.pipe.transformer.context_embedder(edit_t5)
        null_context_embeds = self.pipe.transformer.context_embedder(null_t5)
        
        # Guidance tensor
        guidance = None
        if self.pipe.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=self.device, dtype=torch.float32)
        
        # Create callback to set FluxSpace parameters each step
        def fluxspace_callback(pipe, step_idx, timestep, callback_kwargs):
            # Get joint_attention_kwargs from the pipeline
            jkwargs = pipe._joint_attention_kwargs
            if jkwargs is None:
                return callback_kwargs
            
            # Update step index
            jkwargs["current_timestep_idx"] = step_idx
            
            # Compute temb_edit for this timestep
            timestep_tensor = timestep.expand(1).to(self.dtype)
            if guidance is not None:
                temb_edit = pipe.transformer.time_text_embed(
                    timestep_tensor, guidance * 1000, edited_pooled
                )
            else:
                temb_edit = pipe.transformer.time_text_embed(
                    timestep_tensor, edited_pooled
                )
            
            # Set FluxSpace parameters
            jkwargs["edit_prompt_embeds"] = edit_context_embeds.clone()
            jkwargs["neg_prompt_embeds"] = null_context_embeds.clone()
            jkwargs["temb_edit"] = temb_edit
            
            return callback_kwargs
            
        joint_attention_kwargs = {
            "fluxspace_enabled": True,
            "start_timestep_idx": edit_start_step,
            "stop_timestep_idx": edit_stop_step,
            "edit_content_scale": edit_scale,
            "attention_threshold": attention_threshold,
            "edit_prompt_embeds": edit_context_embeds.clone(),
            "neg_prompt_embeds": null_context_embeds.clone(),
            "temb_edit": None,  # Will be set in callback
            "current_timestep_idx": 0,
        }
        
        # Combine prompt with edit for better results
        combined_prompt = f"{prompt}, {edit_prompt}"
        
        # Run pipeline with FluxSpace
        result = self.pipe(
            image=image,
            prompt=combined_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            joint_attention_kwargs=joint_attention_kwargs,
            pooled_prompt_embeds=edited_pooled,
            callback_on_step_end=fluxspace_callback,
            **kwargs
        )
        
        return result.images[0]


# ============================================================================
# Example Usage
# ============================================================================

def example_basic():
    """Basic FluxSpace editing example"""
    print("\n" + "="*50)
    print("FluxSpace Flux Kontext - Basic Example")
    print("="*50)
    
    # Initialize
    pipe = FluxKontextFluxSpace()
    
    # Load image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    image = load_image(url)
    
    # Edit with FluxSpace
    result = pipe(
        image=image,
        prompt="A photo of a cat",
        edit_prompt="sunglasses",
        edit_scale=3.0,
        edit_global_scale=0.0,
        attention_threshold=0.5,
        seed=42
    )
    
    result.save("output_fluxspace.png")
    print("Saved: output_fluxspace.png")


def example_style_edit():
    """Style transfer with coarse editing"""
    print("\n" + "="*50)
    print("FluxSpace - Style Edit Example")
    print("="*50)
    
    pipe = FluxKontextFluxSpace()
    
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    image = load_image(url)
    
    # Coarse edit for style change
    result = pipe(
        image=image,
        prompt="A photo of a cat",
        edit_prompt="oil painting style",
        edit_scale=1.0,
        edit_global_scale=0.6,  # Higher for global style change
        attention_threshold=0.3,
        seed=42
    )
    
    result.save("output_style.png")
    print("Saved: output_style.png")


def example_combined():
    """Combined fine + coarse editing"""
    print("\n" + "="*50)
    print("FluxSpace - Combined Edit Example")
    print("="*50)
    
    pipe = FluxKontextFluxSpace()
    
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    image = load_image(url)
    
    result = pipe(
        image=image,
        prompt="A photo of a cat",
        edit_prompt="wizard hat, magical fantasy style",
        edit_scale=2.5,
        edit_global_scale=0.4,
        attention_threshold=0.5,
        seed=42
    )
    
    result.save("output_combined.png")
    print("Saved: output_combined.png")


if __name__ == "__main__":
    print("Flux Kontext with FluxSpace")
    print("-" * 40)
    
    # Check for GPU
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. This will be slow on CPU.")
    
    try:
        example_basic()
        # example_style_edit()
        # example_combined()
        print("\nDone!")
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Flux Kontext requires ~24GB VRAM")
