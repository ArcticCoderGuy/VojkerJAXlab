# test_jax.py – Markus Kaprio JAX Jedi valmiustesti
import jax
import jax.numpy as jnp
from jax import random

print("JAX version:", jax.__version__)          # pitäisi olla 0.9.0.1 tai lähellä
print("Devices:", jax.devices())                 # ainakin [CpuDevice(id=0)]

key = random.key(42)                             # determiininen seed → Golden Run -ystävällinen

scalar = jnp.array(3.14159)                      # shape ()
vec    = jnp.arange(5.0)                         # (5,)
col    = vec[:, None]                            # (5,1)
mat    = random.normal(key, (4, 3))              # (4,3)

print(f"scalar: {scalar.shape}")
print(f"vec:    {vec.shape}")
print(f"col:    {col.shape}")
print(f"mat:    {mat.shape}")

# Ensimmäinen shape-contract (Rule #1 manifestosta)
assert mat.shape == (4, 3), "Shape-sopimus rikki!"
print("✅ Kaikki shape-contractit OK – valmis Week 1 Day 1:ään")