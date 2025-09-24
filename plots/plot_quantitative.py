import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


average_diff_times = {
    "none": 0.4227777778,
    "watch": 0.215,
    "dog": 0.08472222222,
}

std_diff_times = {
    "none": 0.09052317076,
    "watch": 0.07788212247,
    "dog": 0.01730446314,
}

average_var_times = {
    "none": 0.04674444444,
    "watch": 0.04805,
    "dog": 0.02601666667,
}

std_var_times = {
    "none": 0.02088647057,
    "watch": 0.01426013455,
    "dog": 0.006787801559,
}


def plot_quantitative(plot_type='diff'):
    """
    Plot either differences or variances based on plot_type flag
    
    Args:
        plot_type (str): 'diff' for differences, 'var' for variances
    """
    
    if plot_type == 'diff':
        # Use difference data
        data = average_diff_times
        errors = std_diff_times
        xlabel = 'Speed Difference (m/s)'
        title = 'Difference from Target Pace'
        filename = 'quantitative_differences.pdf'
    elif plot_type == 'var':
        # Use variance data
        data = average_var_times
        errors = std_var_times
        xlabel = 'Variance (m/s)'
        title = 'Pace Variance'
        filename = 'quantitative_variances.pdf'
    else:
        raise ValueError("plot_type must be 'diff' or 'var'")
    
    # Create the figure and axis
    # fig, ax = plt.subplots(figsize=(5, 3))
    fig, ax = plt.subplots(figsize=(4, 3))

    # Define categories and colors (reversed order)
    categories = ['Dog', 'Watch', 'No Pacer']
    colors = ['mediumseagreen', 'lightblue', 'darkgrey']
    
    # Extract values and errors in the correct order
    values = [data['dog'], data['watch'], data['none']]
    error_values = [errors['dog'], errors['watch'], errors['none']]
    
    # Plot the data
    y_positions = np.arange(len(categories))
    capsize = 2
    error_kw = {'elinewidth': 1, 'capthick': 1, 'ecolor': 'black'}
    ax.barh(y_positions, values, color=colors, xerr=error_values, capsize=capsize, error_kw=error_kw)
    
    # Add labels and title
    plt.xticks(font="Palatino")
    plt.yticks(font="Palatino")

    # this is so we can plot better
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["", "", ""] ) # categories)
    # ax.set_xlabel(xlabel, font="Palatino")
    #
    # ax.set_title(title, font="Palatino", fontsize=14, pad=20)
    
    # For variance plots, use fewer x-axis ticks
    if plot_type == 'var':
        ax.locator_params(axis='x', nbins=4)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save and show the plot
    plt.savefig(filename)
    plt.show()

# Example usage:
plot_quantitative('diff')  # Plot differences
plot_quantitative('var')  # Plot variances

#
# # Sample data
# subcategories = ['Bag 4', 'Bag 3', 'Bag 2', 'Bag 1']
# colors = {
#     'Teach': 'mediumseagreen',
#     'Confirm': 'lightblue',
#     'Async': 'darkgrey',
# }
# data = {
#     'Teach': [95, 54, 16, 8][::-1],
#     'Confirm': [116, 27, 66, 34][::-1],
#     'Async': [73, 121, 139, 116][::-1],
# }
#
# # Convert the data into a format suitable for plotting
# values = np.array([data[cat] for cat in categories])
#
# # Create the figure and axis
# fig, ax = plt.subplots()
#
# # Plot the data
# bottom_values = np.zeros(len(subcategories))
#
# for i, cat in enumerate(categories):
#     ax.barh(subcategories, values[i], label=cat, left=bottom_values, color=colors[cat])
#     bottom_values += values[i]
#
# plt.xticks(font="Palatino")
#
#
# ax.legend()
# plt.yticks(font="Palatino")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# # Add a legend
# ax.legend()
#
# # Show the plot
# plt.show()
