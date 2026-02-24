Python jax.numpy as jnp

# 1. Define the dimensions of our tensor (Pizza example) and with my "Lean JAX Jedi mindset" which is - perfect parts in, perfect parts out, no more, no less. 



  # Always mark-up: batch/row-size (n),  feature dimension (d), FYI -> number of batches (b), output dimension (k) 
  
  # Tensor-Pizza example: n = slices, d = toppings, b = pizzas, k = Output dimension / "hidden" dimension" (e.g., price, rating, etc.)   

n = 8 # every pizza has 8 slices
d = 5 # every slice has 5 toppings (pepperoni, mushroom, onion, olive, pineapple)

print("Process starts - axises defined.")
print (f" n = {n} (slices)")
print (f" d = {d} (tppings)")
  
#2. Creat the pizza tensor X (shape:n, d) 
# Each row = one slice, each column = one topping (amount 0 - 10)

X = jnp.array([
  
  [2,3,0,1,0] # slice 1: cheese 2, Pepperoni 3, Mushroom 0, Onion 1, Olive 0 that are the d = 5 toppings
  [1,0,4,0,2] # slice 2: cheese 1, Pepperoni 0, Mushroom 4, Onion 0, Olive 2
  [0,1,2,3,0] # slice 3: cheese 0, Pepperoni 1, Mushroom 2, Onion 3, Olive 0
  [3,2,1,0,1] # slice 4: cheese 3, Pepperoni 2, Mushroom 1, Onion 0, Olive 1
  [1,4,0,2,3] # slice 5: cheese 1, Pepperoni 4, Mushroom 0, Onion 2, Olive 3
  [0,0,5,1,0] # slice 6: cheese 0, Pepperoni 0, Mushroom 5, Onion 1, Olive 0
  [2,1,3,0,4] # slice 7: cheese 2, Pepperoni 1, Mushroom 3, Onion 0 , Olive 4
   [1,2,1,4 ,2] # slice8: cheese1 , Pepperoni2 , Mushroom1 , Onion4 , Olive2
   
  
]).astype(jnp.float32) # shape (n, d) = (8, 5)


# Make sure you have the f-string and print/assert statements to show/check the shapes of your tensors for good quality code, and that they match the expected shapes based on your definitions.This is a culture-thing and not being stupid, it's about being precise and clear in your code, which is a key part of the JAX Jedi mindset. 

# TL:DR: Don´t get burned by the shape-contracts, always print the shapes and make sure they match your expectations.

print("\n Weighted toppings created -> shape check:")
print(f" X.shape == (n, d), f"cancel ! Pizza should be ({n}, {d}) but got {X.shape}")
print(" -> Shape-contract check passed, (Cpk 3.0 level) ready for the next step: defining the weights and calculating the output tensor Y.")      





