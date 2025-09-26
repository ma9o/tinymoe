from dataclasses import dataclass

@dataclass
class Config:
    # Model
    vocab_size: int = 10_000
    seq_len: int = 512
    n_layers: int = 6
    d_model: int = 256
    n_heads: int = 8
    mlp_type: str = "moe"
    num_experts: int = 4
    k: int = 2
    d_ff_expert: int = 256
    capacity_factor: float = 1.25
    tie_weights: bool = True

    # Training
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    dropout: float = 0.1
    grad_accum_steps: int = 1
    max_steps: int = 2000
    warmup_steps: int = 1000
    log_every: int = 50
    save_every: int = 500

    # Loss coeffs
    aux_loss_coeff: float = 1e-2
    z_loss_coeff: float = 1e-4

    # Data
    dataset_name: str = "roneneldan/TinyStories"
    text_field: str = "text"
    num_workers: int = 2

    # System
    precision: str = "fp16"  # "fp16" or "bf16"
    device: str = "mps"      # MPS only
