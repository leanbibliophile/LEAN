# Adhoc scripts to map downloaded models to our parameters
import torch

def remap_h14(state_dict):
    layers = 32
    hidden_dim = 1280

    mapping = {
        'cls_token': 'class_token',
        'patch_embed.proj.bias': 'patch_embedding.bias', 
        'patch_embed.proj.weight': 'patch_embedding.weight', 
        'pos_embed': 'positional_embedding.pos_embedding',
        'norm.bias': 'norm.bias',
        'norm.weight': 'norm.weight',
        'pre_logits.fc.bias': 'pre_logits.bias',
        'pre_logits.fc.weight': 'pre_logits.weight',
        'head.weight': 'fc.weight',
        'head.bias': 'fc.bias'
    }

    new_state_dict = {}

    for k,v in mapping.items():
        new_state_dict[v] = state_dict[k]

    old_block_prefix = "blocks.{i}"
    new_block_prefix = "transformer.blocks.{i}"
    for index in range(layers):
        old_prefix = old_block_prefix.format(i=index)
        new_prefix = new_block_prefix.format(i=index)

        new_state_dict[f"{new_prefix}.norm1.weight"] = state_dict[f"{old_prefix}.norm1.weight"]
        new_state_dict[f"{new_prefix}.norm1.bias"] = state_dict[f"{old_prefix}.norm1.bias"]
        new_state_dict[f"{new_prefix}.norm2.weight"] = state_dict[f"{old_prefix}.norm2.weight"]
        new_state_dict[f"{new_prefix}.norm2.bias"] = state_dict[f"{old_prefix}.norm2.bias"]

        new_state_dict[f"{new_prefix}.pwff.fc1.weight"] = state_dict[f"{old_prefix}.mlp.fc1.weight"]
        new_state_dict[f"{new_prefix}.pwff.fc1.bias"] = state_dict[f"{old_prefix}.mlp.fc1.bias"]
        new_state_dict[f"{new_prefix}.pwff.fc2.weight"] = state_dict[f"{old_prefix}.mlp.fc2.weight"]
        new_state_dict[f"{new_prefix}.pwff.fc2.bias"] = state_dict[f"{old_prefix}.mlp.fc2.bias"] 

        new_state_dict[f"{new_prefix}.proj.weight"] = state_dict[f"{old_prefix}.attn.proj.weight"]
        new_state_dict[f"{new_prefix}.proj.bias"] = state_dict[f"{old_prefix}.attn.proj.bias"]


        q_w,k_w,v_w = state_dict[f'{old_prefix}.attn.qkv.weight'].split(hidden_dim, dim=0)
        q_b,k_b,v_b = state_dict[f'{old_prefix}.attn.qkv.bias'].split(hidden_dim, dim=0)

        new_state_dict[f'{new_prefix}.attn.proj_q.weight'] = q_w
        new_state_dict[f'{new_prefix}.attn.proj_k.weight'] = k_w
        new_state_dict[f'{new_prefix}.attn.proj_v.weight'] = v_w
        new_state_dict[f'{new_prefix}.attn.proj_q.bias'] = q_b
        new_state_dict[f'{new_prefix}.attn.proj_k.bias'] = k_b
        new_state_dict[f'{new_prefix}.attn.proj_v.bias'] = v_b

    return new_state_dict

if __name__ == "__main__":
    pretrained_path = f"/ws/fs_mount/model_chk/ViT/H_14.pth"
    updated_path = "/ws/fs_mount/model_chk/ViT/H_14_aligned.pth"
    psd = torch.load(pretrained_path)
    new_sd = remap_h14(psd)
    torch.save(new_sd, updated_path)