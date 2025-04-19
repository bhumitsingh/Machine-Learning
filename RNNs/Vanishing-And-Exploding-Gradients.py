import numpy as np
import matplotlib.pyplot as plt

def simulate_gradient_flow(W_val, T=20, hidden_size=1):
    """Simulate the gradient flow for a single weight value."""
    W = np.array([[W_val]])
    h = np.zeros((T + 1, hidden_size))
    h[0] = 1.0 # Initail hidden state

    grad = np.eye(hidden_size) # Initail Gradient is Identity
    grad_norms = []

    for t in range(1, T + 1):
        h[t] = np.tanh(W @ h[t - 1])
        derivative = 1 - h[t] ** 2 # Derivative of tanh
        J = derivative * W # Jacodian
        grad = J @ grad
        grad_norms.append(np.linalg.norm(grad))

    return grad_norms 

# Simulate
timesteps = 20
grad_vanish = simulate_gradient_flow(W_val = 0.5, T=timesteps)
grad_explode = simulate_gradient_flow(W_val = 1.5, T=timesteps)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, timesteps + 1), grad_vanish, label="Vanishing (W=0.5)")
plt.plot(range(1, timesteps + 1), grad_explode, label="Exploding (W=1.5)")
plt.xlabel("Time step")
plt.ylabel("Gradient norm")
plt.title("Vanishing vs Exploding Gradients in RNNs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()