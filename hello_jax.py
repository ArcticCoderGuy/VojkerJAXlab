# hello_jax.py – Markus Kaprio JAX Hello World (2026-02-21)
import jax.numpy as jnp           # JAX:n NumPy-tyylinen paketti (jnp)
import jax                        # itse JAX-kirjasto (käytetään myöhemmin)

# 1. Perus muuttujat ja printtaus (kuten tavallisessa Pythonissa)
nimi = "Markus"
ika = 50                           # (syntymävuosi 2006 → 2026-ikääsi)
tervehdys = f"Hei {nimi}! Olet {ika}-vuotias JAX-Jedi-oppipoika."

print(tervehdys)
print("Tämä on tavallinen Python-string – toimii yhtä hyvin JAX:lla")

# 2. Ensimmäinen JAX-esimerkki: yksinkertainen taulukko (array)
luvut = jnp.array([1, 2, 3, 4, 5])   # luo JAX-taulukko (shape (5,))
print("JAX-taulukko:", luvut)
print("Taulukon shape:", luvut.shape)   # → (5,)
print("Summa:", jnp.sum(luvut))         # → 15
print("Keskiarvo:", jnp.mean(luvut))    # → 3.0

# 3. Pieni laskutoimitus (kuten NumPyssa)
kaksinkertainen = luvut * 2
print("Kaksinkertainen:", kaksinkertainen)   # [2 4 6 8 10]

print("\n✅ Hello JAX World – ensimmäinen askel otettu!")
print("JAX-versio:", jax.__version__)   # pitäisi olla 0.9.0.1