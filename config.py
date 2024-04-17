configurations = {
    "vim-s": {
        "img_size": 224,
        "patch_size": 16,
        "stride": 16,
        "embed_dim": 384,
        "depth": 24,
        "rms_norm": True,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "final_pool_type": "mean",
        "if_abs_pos_embed": True,
        "if_rope": False,
        "if_rope_residual": False,
        "if_cls_token": True,
        "if_devide_out": True,
        "use_middle_cls_token": True,
        "bimamba_type": "v2"
    },
    "vit-s": {
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "num_classes": 2,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "norm_layer": "nn.LayerNorm",
        "eps": 1e-6
    }
}