import matplotlib.pyplot as plt

# Data from the table
models = ['L2-7B', 'L2-13B', 'L3-3B', 'Qwen3-4B', 'Qwen3-8B', 'Gemma3-4B']
h_percent = [72.1, 74.6, 63.4, 57.7, 54.3, 58.9]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('off')  # Hide the axes

# Define table data
table_data = [['LLM'] + models, ['H%'] + [f'{val}%' for val in h_percent]]

# Create the table
table = ax.table(cellText=table_data, loc='center', cellLoc='center')

# Style the table
table.scale(1, 2)  # Scale height of the rows
table.auto_set_font_size(False)
table.set_fontsize(12)

# Make header bold
for key, cell in table.get_celld().items():
    if key[0] == 0 or key[1] == 0:
        cell.set_text_props(weight='bold')

plt.tight_layout()
plt.show()