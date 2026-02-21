# src/jax_arrays.py
# Week 1 Monday – Markus Kaprio – Rule #1: Shapes are a Contract
import jax
import jax.numpy as jnp
from jax import random

def create_shapes(key_seed=42):
    key = random.key(key_seed)  # determiininen seed → Golden Run -ystävällinen

    scalar = jnp.array(3.14159)                  # shape ()
    vec    = jnp.arange(5.0)                     # shape (5,)
    col    = vec[:, None]                        # shape (5, 1)

    # Korjattu: matriisi, jonka viimeinen ulottuvuus = 5 (sopii col:iin)
    mat    = random.normal(key, (4, 5))          # shape (4, 5) ← TÄRKEÄ muutos!

    # Contract-testit (Rule #1)
    assert scalar.shape == (), "Scalar contract failed"
    assert vec.shape == (5,), "Vector contract failed"
    assert col.shape == (5, 1), "Column contract failed"
    assert mat.shape == (4, 5), "Matrix contract failed"

    # Matmul contract – pitäisi onnistua nyt
    result = mat @ col                           # (4,5) @ (5,1) → (4,1)
    assert result.shape == (4, 1), "Matmul contract failed"

    print("scalar:", scalar.shape)
    print("vec:   ", vec.shape)
    print("col:   ", col.shape)
    print("mat:   ", mat.shape)
    print("result:", result.shape)
    print("✅ Kaikki shape-contractit OK – Rule #1 lukittu")

    return {"scalar": scalar, "vec": vec, "col": col, "mat": mat, "result": result}

if __name__ == "__main__":
    create_shapes()