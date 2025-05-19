import numpy as np
import matplotlib.pyplot as plt

# Define the x values to loop through
x_values = [0.3, 0.7]

# Define the range for k
k_values = np.arange(1, 11)  # k values from 1 to 10
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "figure.titlesize": 20,
        "figure.figsize": (10, 5),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "lines.linewidth": 3,
        "lines.markersize": 8,
    }
)
# Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(10, 5))  # Single figure

# Loop through x values and plot for each x
for x in x_values:
    # Calculate the y values for each function
    y1 = (x) ** k_values
    y2 = np.cos(1 - x / 2) ** (2 * k_values)

    # Plot the data with different markers, colors, and labels
    ax.plot(k_values, y1, marker="o", linestyle="-", label="$x^k$ (x=%.2f)" % (x))
    ax.plot(
        k_values,
        y2,
        marker="^",
        linestyle="--",
        label="$\cos^{2k}(1-x/2)$ (x=%.2f)" % (x),
    )

# Customize the plot
ax.set_xlabel("k")
ax.set_ylabel("Function Value")
ax.set_title("Comparison of $x^k$ and $\cos^{2k}(1-x/2)$")
ax.grid(True)
ax.legend()

# Show the plot
plt.tight_layout()
plt.savefig("cosxvsx.png", dpi=300)
plt.savefig("cosxvsx.pdf", dpi=300)
plt.show()
plt.figure()
x = np.linspace(-np.pi / 2, np.pi / 2, 100)
y = np.cos(x)

plt.xticks(
    -np.pi / 2 + np.pi * np.array([0, 0.25, 0.5, 0.75, 1]),
    ["$-\pi/2$", "$-3\pi/4$", "$0$", "$3\pi/4$", "$\pi/2$"],
)
plt.yticks([-1, -0.5, 0, 0.5, 1])

plt.title("$\cos(x)$")
plt.xlabel("x")
plt.ylabel("$\cos(x)$")
plt.grid(True)
plt.tight_layout()
plt.plot(x, y, linewidth=2)
plt.savefig("figures/cosx.png", dpi=300)
plt.savefig("figures/cosx.pdf", dpi=300)
plt.show()