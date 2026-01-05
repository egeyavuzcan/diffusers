"""
FluxSpace Transformer Blocks for Flux Kontext

Adapted from FluxSpace (arXiv:2412.09611) for use with Flux Kontext pipeline.
These blocks replace the original transformer blocks to enable:
- A. Orthogonal Edit Direction Injection
- B. Self-Supervised Spatial Masking
"""

import torch
import math
import torch.nn.functional as F


def apply_rotary_emb(x, freqs_cis, use_real=True, use_real_unbounded_dim=-1):
    """Apply rotary positional embeddings"""
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None].to(x.device)
        sin = sin[None, None].to(x.device)
            
        if use_real_unbounded_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbounded_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


class FluxSpaceTransformerBlock(torch.nn.Module):
    """
    Enhanced Transformer Block with FluxSpace capabilities.
    Wraps the original transformer block and adds:
    - Orthogonal edit direction injection
    - Self-supervised spatial masking
    """
    
    def __init__(self, orig_module, module_name):
        super().__init__()

        self.orig_module = orig_module
        self.module_name = module_name

        # Normalization modules - Latent
        self.norm1 = orig_module.norm1
        self.norm2 = orig_module.norm2

        # Normalization modules - Context
        self.norm1_context = orig_module.norm1_context
        self.norm2_context = orig_module.norm2_context

        # Attention module
        self.attn = orig_module.attn

        # Feed-Forward
        self.ff = orig_module.ff
        self.ff_context = orig_module.ff_context

        # Other hyperparams (may not exist in all versions)
        self._chunk_size = getattr(orig_module, '_chunk_size', None)
        self._chunk_dim = getattr(orig_module, '_chunk_dim', None)

    def calculate_attention_image_text(self, hidden_states, encoder_hidden_states=None, 
                                       attention_mask=None, image_rotary_emb=None):
        """Calculate joint attention between image and text features"""
        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]

        # Predictions from latent
        query = self.attn.to_q(hidden_states)
        key = self.attn.to_k(hidden_states)
        value = self.attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.attn.heads

        query = query.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        if self.attn.norm_q is not None:
            query = self.attn.norm_q(query)
        if self.attn.norm_k is not None:
            key = self.attn.norm_k(key)

        if encoder_hidden_states is not None:
            # Predictions from context
            context_query_proj = self.attn.add_q_proj(encoder_hidden_states)
            context_key_proj = self.attn.add_k_proj(encoder_hidden_states)
            context_value_proj = self.attn.add_v_proj(encoder_hidden_states)

            context_query_proj = context_query_proj.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
            context_key_proj = context_key_proj.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
            context_value_proj = context_value_proj.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

            if self.attn.norm_added_q is not None:
                context_query_proj = self.attn.norm_added_q(context_query_proj)
            if self.attn.norm_added_k is not None:
                context_key_proj = self.attn.norm_added_k(context_key_proj)

            # Combine predictions
            query = torch.cat([context_query_proj, query], dim=2)
            key = torch.cat([context_key_proj, key], dim=2)
            value = torch.cat([context_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        # Compute Attention
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :encoder_hidden_states.shape[1]],  # Context
                hidden_states[:, encoder_hidden_states.shape[1]:]   # Image
            )

            # Projection
            hidden_states = self.attn.to_out[0](hidden_states)
            hidden_states = self.attn.to_out[1](hidden_states)
            encoder_hidden_states = self.attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        
        return hidden_states

    def get_cross_attention_mask(self, image_features, text_features, joint_attention_kwargs):
        """
        B. Self-Supervised Spatial Masking
        Compute attention mask based on cross-attention between image and edit text
        """
        batch_size = image_features.shape[0]

        query = self.attn.to_q(image_features)
        inner_dim = query.shape[-1]
        head_dim = inner_dim // self.attn.heads

        query = query.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        if self.attn.norm_q is not None:
            query = self.attn.norm_q(query)

        key = self.attn.add_k_proj(text_features)
        key = key.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        if self.attn.norm_added_k is not None:
            key = self.attn.norm_added_k(key)

        # Cross-attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = math.sqrt(query.size(-1))
        attention_scores = attention_scores / scale_factor

        # Extract first token attention and average over heads
        attention_map = torch.softmax(attention_scores, dim=-1).mean(dim=1)[:, :, 0]
        
        # Min-max normalization
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-10)
        
        # Sigmoid sharpening (d=10)
        attention_map = F.sigmoid(10 * (attention_map - 0.5)).unsqueeze(-1)
        
        # Thresholding
        threshold = joint_attention_kwargs.get("attention_threshold", 0.5)
        attention_map = (attention_map >= threshold).to(attention_map.dtype)

        return attention_map

    def forward_block(self, hidden_states, encoder_hidden_states, temb, 
                      image_rotary_emb=None, joint_attention_kwargs=None):
        """Standard forward pass without FluxSpace editing"""
        
        # Normalize Latents
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # Normalize Context
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # Attention
        attn_output, context_attn_output = self.calculate_attention_image_text(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb
        )
        
        # Process latent output
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # Process context output
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output
        encoder_hidden_states = encoder_hidden_states + context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward_attention_combine(self, hidden_states, encoder_hidden_states, temb, 
                                  image_rotary_emb=None, joint_attention_kwargs=None):
        """
        FluxSpace forward pass with:
        A. Orthogonal Edit Direction Injection
        B. Self-Supervised Spatial Masking
        """
        edit_encoder_hidden_states = joint_attention_kwargs["edit_prompt_embeds"]
        neg_encoder_hidden_states = joint_attention_kwargs["neg_prompt_embeds"]
        temb_edit = joint_attention_kwargs["temb_edit"]
        edit_content_scale = joint_attention_kwargs["edit_content_scale"]

        # Normalize Latents
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # Normalize Context (base)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb_edit
        )

        # Normalize Edit Context
        norm_edit_hidden_states, c_gate_msa_edit, c_shift_mlp_edit, c_scale_mlp_edit, c_gate_mlp_edit = self.norm1_context(
            edit_encoder_hidden_states, emb=temb_edit
        )

        # Normalize Neg Context
        norm_neg_hidden_states, c_gate_msa_neg, c_shift_mlp_neg, c_scale_mlp_neg, c_gate_mlp_neg = self.norm1_context(
            neg_encoder_hidden_states, emb=temb_edit
        )

        # ==========================================
        # Three parallel attention passes
        # ==========================================
        
        # 1. Base attention
        attn_output, context_attn_output = self.calculate_attention_image_text(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb
        )
        
        # 2. Attention with edit condition
        attn_output_edit, context_attn_output_edit = self.calculate_attention_image_text(
            hidden_states=norm_hidden_states, 
            encoder_hidden_states=norm_edit_hidden_states, 
            image_rotary_emb=image_rotary_emb
        )

        # 3. Attention with null condition (image prior)
        attn_output_neg, context_attn_output_neg = self.calculate_attention_image_text(
            hidden_states=norm_hidden_states, 
            encoder_hidden_states=norm_neg_hidden_states, 
            image_rotary_emb=image_rotary_emb
        )
        
        # ==========================================
        # A. Orthogonal Edit Direction Injection
        # ==========================================
        
        # Orthogonal projection: proj_φ l_θ(x, c_e, t)
        attn_prod = torch.sum(attn_output_edit * attn_output_neg, dim=2, keepdim=True)
        attn_norm_squared = torch.sum(attn_output_neg * attn_output_neg, dim=2, keepdim=True)
        epsilon = 1e-10
        projection = (attn_prod / (attn_norm_squared + epsilon)) * attn_output_neg
        
        # Orthogonal direction: l'_θ(x, c_e, t) = edit - projection
        orthogonal_dir = attn_output_edit - projection

        # ==========================================
        # B. Self-Supervised Spatial Masking
        # ==========================================
        attention_mask = self.get_cross_attention_mask(
            norm_hidden_states, norm_edit_hidden_states, joint_attention_kwargs
        )
        orthogonal_dir = orthogonal_dir * attention_mask

        # ==========================================
        # Apply edit with norm preservation
        # ==========================================
        orig_attn_norm = torch.norm(attn_output, dim=-1, keepdim=True)
        attn_output = attn_output + edit_content_scale * orthogonal_dir
        attn_output = attn_output / (torch.norm(attn_output, dim=-1, keepdim=True) + epsilon) * orig_attn_norm

        # ==========================================
        # Continue with standard processing
        # ==========================================
        
        # Process latent output
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # Process context output
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output
        encoder_hidden_states = encoder_hidden_states + context_ff_output

        # Process edit context output
        context_attn_output_edit = c_gate_msa_edit.unsqueeze(1) * context_attn_output_edit
        edit_encoder_hidden_states = edit_encoder_hidden_states + context_attn_output_edit

        norm_edit_hidden_states = self.norm2_context(edit_encoder_hidden_states)
        norm_edit_hidden_states = norm_edit_hidden_states * (1 + c_scale_mlp_edit[:, None]) + c_shift_mlp_edit[:, None]

        context_edit_ff_output = self.ff_context(norm_edit_hidden_states)
        context_edit_ff_output = c_gate_mlp_edit.unsqueeze(1) * context_edit_ff_output
        edit_encoder_hidden_states = edit_encoder_hidden_states + context_edit_ff_output

        # Process neg context output
        context_attn_output_neg = c_gate_msa_neg.unsqueeze(1) * context_attn_output_neg
        neg_encoder_hidden_states = neg_encoder_hidden_states + context_attn_output_neg

        norm_neg_hidden_states = self.norm2_context(neg_encoder_hidden_states)
        norm_neg_hidden_states = norm_neg_hidden_states * (1 + c_scale_mlp_neg[:, None]) + c_shift_mlp_neg[:, None]

        context_neg_ff_output = self.ff_context(norm_neg_hidden_states)
        context_neg_ff_output = c_gate_mlp_neg.unsqueeze(1) * context_neg_ff_output
        neg_encoder_hidden_states = neg_encoder_hidden_states + context_neg_ff_output

        # Clip for float16 stability
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
            edit_encoder_hidden_states = edit_encoder_hidden_states.clip(-65504, 65504)
            neg_encoder_hidden_states = neg_encoder_hidden_states.clip(-65504, 65504)

        # Update embeddings for next layer
        joint_attention_kwargs["edit_prompt_embeds"] = edit_encoder_hidden_states
        joint_attention_kwargs["neg_prompt_embeds"] = neg_encoder_hidden_states

        return encoder_hidden_states, hidden_states

    def forward(self, hidden_states, encoder_hidden_states, temb, 
                image_rotary_emb=None, joint_attention_kwargs=None):
        """
        Forward pass - automatically switches between standard and FluxSpace mode
        """
        joint_attention_kwargs = joint_attention_kwargs or {}
        
        # Check if FluxSpace editing is enabled and within timestep range
        fluxspace_enabled = joint_attention_kwargs.get("fluxspace_enabled", False)
        current_idx = joint_attention_kwargs.get("current_timestep_idx", 0)
        start_idx = joint_attention_kwargs.get("start_timestep_idx", 0)
        stop_idx = joint_attention_kwargs.get("stop_timestep_idx", float('inf'))
        
        # Also check if required parameters are present
        has_required_params = (
            joint_attention_kwargs.get("edit_prompt_embeds") is not None and
            joint_attention_kwargs.get("neg_prompt_embeds") is not None and
            joint_attention_kwargs.get("temb_edit") is not None
        )
        
        if fluxspace_enabled and has_required_params and start_idx <= current_idx <= stop_idx:
            return self.forward_attention_combine(
                hidden_states, encoder_hidden_states, temb,
                image_rotary_emb=image_rotary_emb, 
                joint_attention_kwargs=joint_attention_kwargs
            )
        else:
            return self.forward_block(
                hidden_states, encoder_hidden_states, temb,
                image_rotary_emb=image_rotary_emb, 
                joint_attention_kwargs=joint_attention_kwargs
            )


class FluxSpaceSingleTransformerBlock(torch.nn.Module):
    """
    Enhanced Single Transformer Block for FluxSpace.
    Single blocks don't have cross-attention, so they use standard forward.
    """
    
    def __init__(self, orig_module, module_name):
        super().__init__()

        self.orig_module = orig_module
        self.module_name = module_name

        # Normalization Module
        self.norm = orig_module.norm
        
        # Projection Modules
        self.mlp_hidden_dim = orig_module.mlp_hidden_dim
        self.proj_mlp = orig_module.proj_mlp
        self.proj_out = orig_module.proj_out
        self.act_mlp = orig_module.act_mlp

        # Attention Module
        self.attn = orig_module.attn

    def calculate_attention(self, hidden_states, attention_mask=None, image_rotary_emb=None):
        """Calculate self-attention"""
        batch_size = hidden_states.shape[0]

        query = self.attn.to_q(hidden_states)
        key = self.attn.to_k(hidden_states)
        value = self.attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.attn.heads

        query = query.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        if self.attn.norm_q is not None:
            query = self.attn.norm_q(query)
        if self.attn.norm_k is not None:
            key = self.attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        return hidden_states

    def forward(self, hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        """Forward pass for single transformer block"""
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.calculate_attention(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb
        )
        
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


# ============================================================================
# Helper functions for module replacement
# ============================================================================

def get_attr(obj, attr):
    """Get nested attribute by dot-separated path"""
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def set_attr_raw(obj, attr, value):
    """Set nested attribute by dot-separated path"""
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)


def register_fluxspace_blocks(transformer):
    """
    Replace transformer blocks with FluxSpace-enhanced versions.
    Call this after loading the pipeline.
    """
    weight_keys = transformer.state_dict().keys()
    transformer_modules = []
    single_transformer_modules = []

    for weight_key in weight_keys:
        module_name = ".".join(weight_key.split(sep=".")[:2])
        if weight_key.startswith("single_transformer_blocks"):
            if module_name not in single_transformer_modules:
                single_transformer_modules.append(module_name)
        elif weight_key.startswith("transformer_blocks"):
            if module_name not in transformer_modules:
                transformer_modules.append(module_name)
    
    # Replace single transformer blocks
    for single_transformer_module in single_transformer_modules:
        orig_module = get_attr(transformer, single_transformer_module)
        unit = FluxSpaceSingleTransformerBlock(orig_module, single_transformer_module)
        set_attr_raw(transformer, single_transformer_module, unit)

    # Replace double transformer blocks
    for transformer_module in transformer_modules:
        orig_module = get_attr(transformer, transformer_module)
        unit = FluxSpaceTransformerBlock(orig_module, transformer_module)
        set_attr_raw(transformer, transformer_module, unit)
    
    return len(transformer_modules), len(single_transformer_modules)
