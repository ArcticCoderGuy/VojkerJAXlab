import jax.numpy as jnp

b = 5
n = 10
d = 3
k = 2

X = jnp.arange(b * n * d).reshape(b, n, d).astype(jnp.float32)
print("Input X shape:", X.shape)

W = jnp.ones((d, k))
print("Painot W shape:", W.shape)

Y = X @ W
print("Tulos Y shape:", Y.shape)

print("\nEnsimmäinen näyte (batch 0, näyte 0):")
print("Input:", X[0, 0, :])
print("Output:", Y[0, 0, :])

print("\nValmis! Shape-contract pidetty.")



