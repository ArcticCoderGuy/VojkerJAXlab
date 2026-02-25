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
   [1,2,1,4,2] # slice8: cheese1 , Pepperoni2 , Mushroom1 , Onion4 , Olive2
   
  
]).astype(jnp.float32) # shape (n, d) = (8, 5)


# Make sure you have the f-string and print/assert statements to show/check the shapes of your tensors for good quality code, and that they match the expected shapes based on your definitions.This is a culture-thing and not being stupid, it's about being precise and clear in your code, which is a key part of the JAX Jedi mindset. 

# TL:DR: Don´t get burned by the shape-contracts, always print the shapes and make sure they match your expectations.

print("\n Weighted toppings created -> shape check:")
print(f" X.shape == (n, d), f"cancel ! Pizza should be ({n}, {d}) but got {X.shape}")
print(" -> Shape-contract check passed, (Cpk 3.0 level) ready for the next step: defining the weights and calculating the output tensor Y.")      

# We shot the first slice to understand the shape-contracts
print("first slice (line 0):", X[0, :])

#3. Create/define the weights W (shape: d, 1)- each topping get´s it´s own weighted value-rating
W = jnp.array([1.5, 2.0, 0.5, 1.0, 0.8])[:, None] # Cheese has the most weight, then pepperoni, then onion, then olive and mushroom has no weight at all. Shape (d, 1) = (5, 1)

#Cpk 3.0: Print + Check the shape-contracts for the weights W
print("\n Weights defined -> shape check:")
print(" W.shaope =",W.shape)
assert W.shape == (d, 1),f"Cancel ! Weights should be ({d}, 1) but got {W.shape}"
print(" ->Shape OK ! (Cpk 3.0-level)")


# 4. Calculate the output tensor Y (shape: n, 1) = (8, 1) - each slice gets a weighted score based on the toppings it has and their weights. This is a simple matrix multiplication (dot product) between X and W.

#TL:DR We use the X @ W notation for matrix multiplication, which is a common and efficient way to calculate the output tensor Y in JAX. This operation will take each slice (row) of X, multiply it by the corresponding weights in W, and sum up the results to give us a single score for each slice.

flavor_pointer = X @ W # Shape (n, d) @ (d, 1) = (n, 1)

# Cpk 3.0: Print + Check the shape-contracts for the output tensor 
print("\n flavor_pointer calculated -> shape check:")
print(" flavor_pointer.shape =", flavor_pointer.shape")
assert flavor_pointer.shape == (n, 1), f"Cancel ! Output should be ({n}, 1) but got {flavor_pointer.shape}"
print(" -> Shape-contract check passed, (Cpk 3.0 level) ready for the next step: interpreting the results and maybe doing some more calculations based on the output tensor Y.")


#5 Show concrete results: print the flavor pointer for each slice, which is the weighted score based on the toppings and their weights. This will give us an idea of which slices are more flavorful based on the defined weights.

print("\nFlavor pointer for each slice:")
print("toppings", X[0, :])
print("flavor_pointer", flavor_pointer[0, :])

# End of Code - process accepted 

print(" \n ✅ Pizza Tensor example completed successfully! All shape-contracts held, and the flavor pointer has been calculated for each slice. Ready for the next step in our JAX journey: maybe adding more features, calculating gradients, or even building a simple neural network to predict pizza ratings based on the toppings! Stay tuned for more JAX adventures. \n")

print(" Perfect parts in, perfect parts out - the JAX Jedi mindset in action! Remember to always check your shape-contracts and print your tensors to ensure everything is as expected. This is key to writing clean and efficient JAX code. Until next time, may the JAX be with you! \n")

print( "No canceling this pizza, it is a perfect slice of JAX code! 🍕✨")



