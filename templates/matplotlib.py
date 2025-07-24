import matplotlib.pyplot as plt

# Sample data
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 7, 12]

# Create a figure with two subplots: Bar graph and Dot plot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# --- Bar Graph ---
axs[0].bar(categories, values, color='skyblue')
axs[0].set_title('Bar Graph')
axs[0].set_xlabel('Category')
axs[0].set_ylabel('Value')

# --- Dot Plot ---
axs[1].scatter(categories, values, color='red', s=100)  # s is marker size
axs[1].set_title('Dot Plot')
axs[1].set_xlabel('Category')
axs[1].set_ylabel('Value')

# Adjust layout and display
plt.tight_layout()
plt.show()
