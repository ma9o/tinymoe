**KV Cache**

- ✅ Cache: K and V for all past tokens
- ❌ Don't cache: Q (unique per token, used once) or attention scores (recomputed every time)
- Each new token: compute fresh Q_i, attend to all cached K's and V's
- Saves recomputing the projection matrices W_k and W_v on all past tokens! (tradeoff memory for speed)

**Attention heads weight sharing**

MHA (Multi-Head Attention), GQA (Grouped Query Attention), MQA (Multi-Query Attention) are schemes for sharing K and V projection weights across attention heads. They save memory but we still do the same amount of ops due to broadcasting back to (b, t, kv * self.group_size, d).

- MHA: n_kv_heads = n_heads (12 K, 12 V)
- GQA: n_kv_heads < n_heads (4 K, 4 V shared across groups)
- MQA: n_kv_heads = 1 (1 K, 1 V shared by all)


**Cross-Attention**

```python
Q = W_q @ X        # X is your target/decoder sequence
K = W_k @ context  # context is from encoder/different source
V = W_v @ context  # Same context source
```

**Biases and Normalization**

LayerNorm Undoes:
- Shift: Subtracting mean removes translation/bias
- Scale: Dividing by std deviation normalizes magnitude
- Does NOT remove rotation/mixing from weight matrices in FFN (or softmax non-linearity in attention)
- That's why LayerNorm includes learnable γ (scale) and β (shift) parameters

Attention bias still matters because it affects relative weighting through softmax, not absolute output values (unlike output bias which gets normalized away. and gets removed in modern architectures).

RMSNorm only removes scale, not shift but is much faster (one pass vs two passes. in theory it should be affected by bias but in practice it works fine - likely it wasn;t doing much anyway).

```
mean = sum(x) / n              # 1 pass
var = sum((x - mean)²) / n     # 2 passes
y = (x - mean) / sqrt(var) * γ + β
---
rms = sqrt(sum(x²) / n) # 1 pass
y = x / rms * γ
```