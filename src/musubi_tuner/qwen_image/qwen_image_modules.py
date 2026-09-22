from torch import nn

# region activations


ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")


# endregion activations


# Attention extracted from Musubi Tuner's hunyuan_model/attention.py.
# Preserve the original Qwen-used SDPA, FlashAttention and xformers computations.
import torch
import torch.nn.functional as F

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_func
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_func = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

MEMORY_LAYOUT = {
    "flash": (lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]), lambda x: x),
    "flash_fixlen": (lambda x: x, lambda x: x),
    "torch": (lambda x: x.transpose(1, 2), lambda x: x.transpose(1, 2)),
    "xformers": (lambda x: x, lambda x: x),
}


def attention(
    q_or_qkv_list,
    k=None,
    v=None,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    total_len=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    q, k, v = q_or_qkv_list if type(q_or_qkv_list) == list else (q_or_qkv_list, k, v)
    if type(q_or_qkv_list) == list:
        q_or_qkv_list.clear()
    if mode == "sdpa":
        mode = "torch"
    split_attn = total_len is not None
    if (split_attn or cu_seqlens_q is None) and mode == "flash":
        mode = "flash_fixlen"
    # print(f"Attention mode: {mode}, split_attn: {split_attn}")
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]

    # trim the sequence length to the actual length instead of attn_mask
    if split_attn:
        trimmed_len = q.shape[1] - total_len
        q = [q[i : i + 1, : total_len[i]] for i in range(len(q))]
        k = [k[i : i + 1, : total_len[i]] for i in range(len(k))]
        v = [v[i : i + 1, : total_len[i]] for i in range(len(v))]
        q = [pre_attn_layout(q_i) for q_i in q]
        k = [pre_attn_layout(k_i) for k_i in k]
        v = [pre_attn_layout(v_i) for v_i in v]
        # print(
        #     f"Trimming the sequence length to {total_len},trimmed_len: {trimmed_len}, q.shape: {[q_i.shape for q_i in q]}, mode: {mode}"
        # )
    else:
        q = pre_attn_layout(q)
        k = pre_attn_layout(k)
        v = pre_attn_layout(v)

    if mode == "torch":
        if split_attn:
            x = []
            for i in range(len(q)):
                x_i = F.scaled_dot_product_attention(q[i], k[i], v[i], dropout_p=drop_rate, is_causal=causal)
                q[i], k[i], v[i] = None, None, None
                x.append(x_i)
            del q, k, v
        else:
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(q.dtype)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)
            del q, k, v
            del attn_mask

    elif mode == "xformers":
        # B, M, H, K: M is the sequence length, H is the number of heads, K is the dimension of the heads -> it is same as input dimension
        # currently only support batch_size = 1
        assert split_attn or cu_seqlens_q is None, "Xformers only supports splitting"
        if split_attn:
            x = []
            for i in range(len(q)):
                x_i = xops.memory_efficient_attention(q[i], k[i], v[i], p=drop_rate)  # , causal=causal)
                q[i], k[i], v[i] = None, None, None
                x.append(x_i)
            del q, k, v
        else:
            x = xops.memory_efficient_attention(q, k, v, p=drop_rate)
            del q, k, v

    elif mode == "flash":
        x = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        del q, k, v
        # x with shape [(bxs), a, d]
        x = x.view(batch_size, max_seqlen_q, x.shape[-2], x.shape[-1])  # reshape x to [b, s, a, d]

    elif mode == "flash_fixlen":
        if split_attn:
            x = []
            for i in range(len(q)):
                # q: (batch_size, seqlen, nheads, headdim), k: (batch_size, seqlen, nheads_k, headdim), v: (batch_size, seqlen, nheads_k, headdim)
                x_i = flash_attn_func(q[i], k[i], v[i], dropout_p=drop_rate, causal=causal)
                q[i], k[i], v[i] = None, None, None
                x.append(x_i)
            del q, k, v
        else:
            x = flash_attn_func(q, k, v, dropout_p=drop_rate, causal=causal)
            del q, k, v  # this causes error in compiled mode with fullgraph=True

    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    if split_attn:
        x = [post_attn_layout(x_i) for x_i in x]
        for i in range(len(x)):
            x[i] = F.pad(x[i], (0, 0, 0, 0, 0, trimmed_len[i]))
        x = torch.cat(x, dim=0)
    else:
        x = post_attn_layout(x)

    b, s, a, d = x.shape
    x = x.reshape(b, s, -1)
    return x
