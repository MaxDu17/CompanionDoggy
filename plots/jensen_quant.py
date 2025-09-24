import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Jensen's speed data (m/s)
jensen_speeds = [1.180244186, 1.727868852, 1.837604563, 1.92775, 1.65]
jensen_distances = [
    2030.02,
    2108,
    2416.45,
    2467.52,
    2024.74
]
jensen_total_distances = [sum(jensen_distances[:i+1]) for i in range(len(jensen_distances))]

# Create time points for the speeds (assuming 5-second intervals)
time_points = np.arange(1, len(jensen_speeds)+1, 1)  # 0, 5, 10, 15 seconds

# Flag to choose what to plot
plot_speed = True  # Set to False to plot total distance instead

print("Jensen's speeds:", jensen_speeds)
print("Jensen's total distances:", jensen_total_distances)
print("Time points:", time_points)

def plot_jensen_data():
    """
    Plot Jensen's speed or total distance data over time based on flag
    """
    # Create the figure and axis
    # fig, ax = plt.subplots(figsize=(4, 5))
    fig, ax = plt.subplots(figsize=(3, 3))

    # Choose data to plot based on flag
    if plot_speed:
        data = jensen_speeds
        ylabel = 'Speed (m/s)'
        title = "Jensen's Speed Over Time"
        filename = 'jensen_speeds.pdf'
        # Set y-axis ticks for speed
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
        ax.set_ylim(1.0, 2.0)
        ax.set_yticklabels([0, 0.5, 1.0, 1.5, 2.0, 2.5], font="Palatino")
    else:
        data = [x / 1000 for x in jensen_total_distances]
        ylabel = 'Total Distance (km)'
        title = "Jensen's Total Distance Over Time"
        filename = 'jensen_distances.pdf'
        # Set y-axis ticks for distance (auto-scale)
        ax.set_yticks([2, 4, 6, 8, 10, 12])
        ax.set_yticklabels([2, 4, 6, 8, 10, 12], font="Palatino")
        ax.set_ylim(0, 12)

        # ax.tick_params(axis='y', labelsize=10)
    
    # Plot the data
    ax.plot(time_points, data, 'o-', color='mediumseagreen', linewidth=2, 
           markersize=8, label="Jensen's Data")
    
    # Add labels and title
    # ax.set_xlabel('Session Number', fontsize=12, font="Palatino")
    # ax.set_ylabel(ylabel, fontsize=12, font="Palatino")
    # ax.set_title(title, fontsize=14, font="Palatino", pad=20)
    
    # Set x-axis ticks
    ax.set_xticks(time_points)
    ax.set_xticklabels(time_points, font="Palatino")
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Change y-axis tick label font size
    ax.tick_params(axis="y", labelsize=12)

    # Change x-axis tick label font size
    ax.tick_params(axis="x", labelsize=12)
    
    # Add grid
    # ax.grid(True, alpha=0.3)
    
    # Add legend
    # ax.legend(frameon=False, loc='upper right', fontfamily="Palatino")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show
    plt.savefig(filename)
    plt.show()

# Run the plot
plot_jensen_data()
